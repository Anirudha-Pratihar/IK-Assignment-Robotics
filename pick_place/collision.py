"""Signed-distance collision handling shared by the planner, the QP and DLS steppers (mink has its own, equivalent, limit).
 Groups hold (robot geoms, obstacle geoms, d_min).  Per tick we query mj_geomDistance only for pairs closer than d_detect and build
   * G dq <= h   rows  (QP)          :  d + n.(J2-J1) dq dt >= d_min      (same math as mink.CollisionAvoidanceLimit)
   * a repulsive joint velocity (DLS):  potential-field push along the contact normal when d < d_min + band
   * clearance metrics (min signed distance)"""
from dataclasses import dataclass, field
import numpy as np
import mujoco
from .scene import geom_ids_under, colliding_geoms_of_bodies, pad_geom_ids
from .robots import get_robot


@dataclass
class Group:
    name: str
    g1: list            # robot-side geom ids
    g2: list            # obstacle geom ids
    d_min: float
    band: float = 0.015      # DLS repulsion band (active for d < d_min + band)
    pairs: list = field(default_factory=list)
    def __post_init__(self):
        self.pairs = [(a, b) for a in self.g1 for b in self.g2]


class CollisionSets:
    """Builds the named geom groups from the PLAN model and returns the groups active in each phase."""
    def __init__(self, scene, cfg):
        m = scene.model; self.m = m; self.cfg = cfg
        gid = lambda n: m.geom(n).id
        names = [m.geom(i).name for i in range(m.ngeom)]
        rs = get_robot(cfg.robot)
        carried = gid("carried_cube")
        arm = [g for g in colliding_geoms_of_bodies(m, rs.env_bodies) if g != carried]
        hand_all = geom_ids_under(m, scene.ee_body)
        pads = pad_geom_ids(m, scene.gripper)
        hand = [g for g in hand_all if g not in pads and g not in arm and g != carried and (m.geom_contype[g] or m.geom_conaffinity[g])]
        table = [gid("table_top")] + [gid(f"table_leg{i}") for i in range(4)]
        tray = [gid(n) for n in ("tray_base", "tray_wall_px", "tray_wall_nx", "tray_wall_py", "tray_wall_ny")]
        cube = gid("cube_geom")
        dm = cfg.d_min
        self.ids = dict(arm=arm, hand=hand, pads=pads, carried=carried, table=table, tray=tray, cube=cube)
        self.G = dict(
            robot_env=Group("robot_env", arm + hand, table + tray, dm),
            pads_table=Group("pads_table", pads, table, 0.006, 0.004),
            pads_tray=Group("pads_tray", pads, tray, dm, cfg.pads_tray_band),
            pads_cube=Group("pads_cube", pads + hand, [cube], 0.006, 0.004),
            carried_tray=Group("carried_tray", [carried], tray, dm, 0.005),
            carried_table=Group("carried_table", [carried], table, dm, 0.005),
        )
        # which groups are active in which phase.  (carried cube vs table is skipped in 'lift' because the cube starts ON the table)
        self.phase_groups = {
            "approach": ["robot_env", "pads_table", "pads_tray", "pads_cube"],
            "descend":  ["robot_env", "pads_table", "pads_tray", "pads_cube"],
            "lift":     ["robot_env", "pads_table", "pads_tray", "carried_tray"],
            "transit":  ["robot_env", "pads_table", "pads_tray", "carried_tray", "carried_table"],
            "place":    ["robot_env", "pads_table", "pads_tray", "carried_tray", "carried_table"],
            "retreat":  ["robot_env", "pads_table", "pads_tray"],
        }

    def groups(self, phase, enabled=True):
        return [self.G[n] for n in self.phase_groups.get(phase, [])] if enabled else []


def _dist(m, d, g1, g2, dmax, fromto):
    return mujoco.mj_geomDistance(m, d, g1, g2, dmax, fromto)


def min_clearance(m, d, groups, dmax=0.3):
    """smallest signed distance over all pairs of the given groups (kinematics must be up to date)"""
    best = dmax; fromto = np.zeros(6)
    for gr in groups:
        for a, b in gr.pairs:
            best = min(best, _dist(m, d, a, b, dmax, fromto))
    return best


def qp_rows(prob, groups, dt, d_detect, gain=0.85):
    """inequality rows (G, h) with G dq <= h on the ARM joint velocities"""
    m, d = prob.model, prob.data
    rows, rhs = [], []
    fromto = np.zeros(6); j1 = np.zeros((3, m.nv)); j2 = np.zeros((3, m.nv))
    for gr in groups:
        for a, b in gr.pairs:
            dist = _dist(m, d, a, b, d_detect, fromto)
            if dist >= d_detect: continue
            n = fromto[3:] - fromto[:3]; nn = np.linalg.norm(n)
            if nn < 1e-9: continue
            n = n / nn
            mujoco.mj_jac(m, d, j1, None, fromto[:3], m.geom_bodyid[a])
            mujoco.mj_jac(m, d, j2, None, fromto[3:], m.geom_bodyid[b])
            row = n @ (j2 - j1)                               # d(dist)/dt = row @ dq
            rows.append(-row[prob.dofs]); rhs.append(gain * (dist - gr.d_min) / dt)
    if not rows: return None, None
    return np.array(rows), np.array(rhs)


def repulsion_dq(prob, groups, d_detect, v_max=0.25):
    """DLS-side collision handling: potential field.  Active only for d < d_min + band."""
    m, d = prob.model, prob.data
    dq = np.zeros(prob.n); fromto = np.zeros(6); j1 = np.zeros((3, m.nv)); j2 = np.zeros((3, m.nv)); n_act = 0
    for gr in groups:
        for a, b in gr.pairs:
            band = gr.band
            dist = _dist(m, d, a, b, gr.d_min + band, fromto)
            if dist >= gr.d_min + band: continue
            n = fromto[3:] - fromto[:3]; nn = np.linalg.norm(n)
            if nn < 1e-9: continue
            n = n / nn
            mujoco.mj_jac(m, d, j1, None, fromto[:3], m.geom_bodyid[a])
            mujoco.mj_jac(m, d, j2, None, fromto[3:], m.geom_bodyid[b])
            row = (n @ (j2 - j1))[prob.dofs]
            speed = v_max * min(1.0, (gr.d_min + band - dist) / band)
            dq += row * speed / (row @ row + 1e-6); n_act += 1
    return dq, n_act
