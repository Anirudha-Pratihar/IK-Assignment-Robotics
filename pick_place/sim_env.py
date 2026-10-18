"""Physics side: position-servo arm with gravity + damping feed-forward, gripper commands, contact classification, clearance."""
import numpy as np
import mujoco
from .config import Config
from .scene import build_scene, geom_ids_under, colliding_geoms_of_bodies, pad_geom_ids
from .robots import get_robot
from .collision import CollisionSets, min_clearance


class SimEnv:
    def __init__(self, cfg: Config, design: dict):
        self.cfg, self.design = cfg, design
        self.sc = build_scene(cfg, design, for_planning=False)
        self.model = self.sc.model; self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = int(round(cfg.ik_dt / m.opt.timestep))
        self.damp = m.dof_damping[self.sc.arm_dofs].copy()
        gid = lambda n: m.geom(n).id
        self.id_cube = gid("cube_geom")
        rs = get_robot(cfg.robot)
        self.arm_g = set(colliding_geoms_of_bodies(m, rs.arm_bodies))
        self.arm_env_g = set(colliding_geoms_of_bodies(m, rs.env_bodies))      # arm part used for clearance metrics
        hand_all = geom_ids_under(m, self.sc.ee_body)
        self.pad_g = set(pad_geom_ids(m, self.sc.gripper))
        self.hand_g = set(g for g in hand_all if g not in self.pad_g and g not in self.arm_g and (m.geom_contype[g] or m.geom_conaffinity[g]))
        self.n_arm = cfg.n_arm
        self.table_g = set([gid("table_top")] + [gid(f"table_leg{i}") for i in range(4)])
        self.tray_g = set(gid(n) for n in ("tray_base", "tray_wall_px", "tray_wall_nx", "tray_wall_py", "tray_wall_ny"))
        self.floor_g = {gid("floor")}
        self.env_g = self.table_g | self.tray_g | self.floor_g
        self.frames = []; self.recorder = None
        self.H = design["H"]; self.s = cfg.cube_size / 2

    # ---------------- episode reset ----------------
    def reset(self, q_home, cube_xy, cube_yaw):
        m, d, sc = self.model, self.data, self.sc
        mujoco.mj_resetData(m, d)
        d.qpos[sc.arm_qadr] = q_home
        d.qpos[sc.cube_qadr:sc.cube_qadr + 3] = [cube_xy[0], cube_xy[1], self.H + self.s + 5e-4]
        d.qpos[sc.cube_qadr + 3:sc.cube_qadr + 7] = [np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)]
        d.ctrl[sc.arm_act] = q_home; d.ctrl[sc.grip_act] = sc.gripper.open_cmd
        mujoco.mj_forward(m, d)
        self.q_ref = np.array(q_home, float); self.dq_ref = np.zeros(self.n_arm)
        for _ in range(int(0.4 / m.opt.timestep)): self.step()     # let cube settle, servo converge

    # ---------------- control ----------------
    def q(self): return self.data.qpos[self.sc.arm_qadr].copy()

    def step(self, q_ref=None, dq_ref=None):
        d, sc = self.data, self.sc
        if q_ref is not None: self.q_ref = q_ref
        if dq_ref is not None: self.dq_ref = dq_ref
        d.ctrl[sc.arm_act] = self.q_ref
        d.qfrc_applied[sc.arm_dofs] = d.qfrc_bias[sc.arm_dofs] + self.damp * self.dq_ref     # gravity + damping feed-forward
        mujoco.mj_step(self.model, d)

    def set_gripper(self, close: bool):
        g = self.sc.gripper
        self.data.ctrl[self.sc.grip_act] = g.close_cmd if close else g.open_cmd

    def hold(self, seconds, monitor=None):
        for i in range(int(seconds / self.model.opt.timestep)):
            self.step(self.q_ref, np.zeros(self.n_arm))
            if monitor and monitor(): return False
            if self.recorder and i % self.n_sub == 0: self.recorder.maybe_capture(self)
        return True

    def run_trajectory(self, Q, dQ, monitor=None, tick_cb=None):
        """Q, dQ: (n,6) planned joint positions / velocities at ik_dt spacing.  Returns False if the monitor aborted."""
        q_prev = self.q_ref.copy()
        for k in range(len(Q)):
            for j in range(self.n_sub):
                a = (j + 1) / self.n_sub
                self.step(q_prev * (1 - a) + Q[k] * a, dQ[k])
                if monitor and monitor(): return False
            q_prev = Q[k]
            if tick_cb: tick_cb(k)
            if self.recorder: self.recorder.maybe_capture(self)
        return True

    # ---------------- observations ----------------
    def tcp(self):
        d, sc = self.data, self.sc
        R = d.site_xmat[sc.tcp_site].reshape(3, 3).copy(); return d.site_xpos[sc.tcp_site].copy(), R

    def cube_pos(self): return self.data.qpos[self.sc.cube_qadr:self.sc.cube_qadr + 3].copy()
    def cube_vel(self): return float(np.linalg.norm(self.data.qvel[-6:-3]))      # cube free joint is the last joint

    def finger_travel(self):
        """mean finger slide/driver angle - 0 open; larger = more closed (used to detect 'closed on nothing')"""
        m = self.model; adr = [m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_type[j] in (2, 3) and "gripper" in m.joint(j).name
                               and ("slide" in m.joint(j).name or "driver" in m.joint(j).name)]
        return float(np.mean(self.data.qpos[adr])) if adr else 0.0

    # ---------------- contacts ----------------
    def contact_violations(self, phase):
        """list of (kind, geomA, geomB) for contacts that are NOT part of the nominal task in this phase"""
        d, m = self.data, self.model; out = []
        carrying = phase in ("lift", "transit", "place")
        for i in range(d.ncon):
            c = d.contact[i]; a, b = int(c.geom1), int(c.geom2)
            for x, y in ((a, b), (b, a)):
                x_robot = x in self.arm_g or x in self.hand_g or x in self.pad_g
                if x_robot and y in self.env_g: out.append(("robot_env", a, b)); break
                if (x in self.arm_g or x in self.hand_g) and y == self.id_cube: out.append(("robot_cube", a, b)); break
                if x in self.arm_g and y in self.arm_g: out.append(("self", a, b)); break
                if x == self.id_cube and y in self.tray_g and carrying: out.append(("cube_tray", a, b)); break
                if x == self.id_cube and y in self.table_g and phase in ("transit", "place"): out.append(("cube_table", a, b)); break
        return out

    def name_pair(self, a, b):
        m = self.model; f = lambda g: (m.geom(g).name or f"{m.body(m.geom_bodyid[g]).name}#{g}")
        return f"{f(a)}|{f(b)}"

    def cube_pad_contacts(self):
        sides = set()          # count distinct pad BODIES (one per jaw) - a built-in hand has several pad geoms per finger
        for i in range(self.data.ncon):
            c = self.data.contact[i]; a, b = int(c.geom1), int(c.geom2)
            if self.id_cube in (a, b):
                o = b if a == self.id_cube else a
                if o in self.pad_g: sides.add(int(self.model.geom_bodyid[o]))
        return len(sides)

    def robot_env_clearance(self, groups):
        mujoco.mj_kinematics(self.model, self.data)
        return min_clearance(self.model, self.data, groups)
