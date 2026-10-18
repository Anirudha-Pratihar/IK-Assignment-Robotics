"""Scene builder (MjSpec).  Two compiled variants of the SAME world:
   * sim model  : what physics runs on.
   * plan model : identical kinematics + a 'carried_cube' geom welded to the hand (so every IK method can
                  keep the cube a safe distance from the tray/table while it is being carried).
   qpos layout (both): [arm joints 0..5][gripper joints][cube freejoint x y z qw qx qy qz]."""
from dataclasses import dataclass
import numpy as np
import mujoco
from .config import Config
from .gripper import make_gripper, GripperInfo
from .robots import get_robot
WOOD = [0.80, 0.60, 0.40, 1]


@dataclass
class Scene:
    cfg: Config
    model: mujoco.MjModel
    gripper: GripperInfo
    design: dict
    for_planning: bool

    # filled by finalize()
    def finalize(self):
        m = self.model
        n = self.cfg.n_arm
        self.arm_jids = np.arange(n)
        self.arm_qadr = m.jnt_qposadr[self.arm_jids]
        self.arm_dofs = m.jnt_dofadr[self.arm_jids]
        self.arm_act = np.array([m.actuator(a).id for a in get_robot(self.cfg.robot).arm_actuators])
        self.grip_act = m.actuator(self.gripper.actuator).id
        self.tcp_site = m.site("tcp").id
        self.ee_body = m.body(self.cfg.ee_body).id
        if "cube" in [m.body(i).name for i in range(m.nbody)]:
            self.cube_body = m.body("cube").id
            self.cube_qadr = m.jnt_qposadr[m.joint("cube_joint").id]
            self.cube_geom = m.geom("cube_geom").id
        self.qmin, self.qmax = m.jnt_range[self.arm_jids].T.copy()
        return self


def _box(body, name, size, pos, rgba, contype=1, conaffinity=1, friction=(1.0, 0.01, 0.001), **kw):
    return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=list(pos),
                         rgba=list(rgba), contype=contype, conaffinity=conaffinity, friction=list(friction), **kw)


def build_scene(cfg: Config, design: dict | None = None, for_planning=False, with_cube=True) -> Scene:
    spec = mujoco.MjSpec.from_file(cfg.arm_xml)
    spec.option.timestep = cfg.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 1920; spec.visual.global_.offheight = 1080      # allow big offscreen renders (videos)
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO
    # the HEAL xml ships a fully transparent 'arm' material -> make links visible
    rs = get_robot(cfg.robot)
    if rs.material_fix:
        try: spec.material(rs.material_fix).rgba = [0.72, 0.72, 0.74, 1]
        except Exception: pass

    # ---- arm: HEAL has raw torque motors -> position servos (kp,kv); Franka already has position servos (kept).
    #      Gravity is compensated in SimEnv for both ----
    if rs.replace_servo:
        for name in rs.arm_actuators:
            a = spec.actuator(name)
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = cfg.servo_kp
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[:3] = [0.0, -cfg.servo_kp, -cfg.servo_kv]

    if cfg.limit_scale != 1.0:
        for jn in rs.arm_joints:
            j = spec.joint(jn); lo, hi = j.range; c, h = (lo + hi) / 2, (hi - lo) / 2 * cfg.limit_scale
            j.range = [c - h, c + h]

    # ---- gripper ----
    grip = make_gripper(cfg)
    if rs.gripper_mode == "attach":
        spec.attach(grip.spec, prefix=grip.prefix, site=spec.site(cfg.tool_site))
    else:                                    # built-in hand: stiffen the finger servo (stock: kp=100 N/m -> ~1 N per finger)
        a = spec.actuator(grip.actuator); k = cfg.franka_grip_kp
        a.gainprm[0] = 0.01568627451 * k / 100.0; a.biasprm[1] = -k; a.biasprm[2] = -k / 10.0
    if rs.pedestal is not None:              # visual column under a raised base (no collision)
        px, py, pz = rs.pedestal
        spec.worldbody.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.09, 0.09, pz / 2 - 0.01], pos=[px, py, pz / 2 - 0.01],
                                rgba=[0.3, 0.3, 0.33, 1], contype=0, conaffinity=0)
    ee = spec.body(cfg.ee_body)
    ee.add_site(name="tcp", pos=[0, 0, grip.tcp_z], size=[0.008, 0.008, 0.008], rgba=[0, 1, 0, 1], group=2)

    # ---- environment (only if a design exists; the workspace analysis runs without it) ----
    wb = spec.worldbody
    ct = 9 if for_planning else 1         # plan model: bit 8 lets the carried-cube geom 'see' table/tray in distance queries
    if design is not None:
        H, L, W = design["H"], design["L"], design["W"]
        cx, cy = design["cx"], design["cy"]
        tt = cfg.table_thickness
        table = wb.add_body(name="table", pos=[0, 0, 0])
        _box(table, "table_top", [L / 2, W / 2, tt / 2], [cx, cy, H - tt / 2], WOOD, ct, ct, friction=(1.0, 0.01, 0.001))
        for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
            _box(table, f"table_leg{i}", [0.02, 0.02, (H - tt) / 2],
                 [cx + sx * (L / 2 - 0.03), cy + sy * (W / 2 - 0.03), (H - tt) / 2], [0.35, 0.3, 0.25, 1], ct, ct)
        # tray (open box) sitting on the table
        tx, ty = design["tray_center"]; ix, iy = cfg.tray_inner; t, hb, hw = cfg.tray_wall_t, cfg.tray_base_t, cfg.tray_wall_h
        tray = wb.add_body(name="tray", pos=[tx, ty, H])
        dark = [0.1, 0.1, 0.12, 1]
        _box(tray, "tray_base", [ix / 2 + t, iy / 2 + t, hb / 2], [0, 0, hb / 2], dark, ct, ct, friction=(1.0, 0.01, 0.001))
        _box(tray, "tray_wall_px", [t / 2, iy / 2 + t, hw / 2], [ix / 2 + t / 2, 0, hw / 2], dark, ct, ct)
        _box(tray, "tray_wall_nx", [t / 2, iy / 2 + t, hw / 2], [-ix / 2 - t / 2, 0, hw / 2], dark, ct, ct)
        _box(tray, "tray_wall_py", [ix / 2, t / 2, hw / 2], [0, iy / 2 + t / 2, hw / 2], dark, ct, ct)
        _box(tray, "tray_wall_ny", [ix / 2, t / 2, hw / 2], [0, -iy / 2 - t / 2, hw / 2], dark, ct, ct)
        if with_cube:
            s = cfg.cube_size / 2
            cube = wb.add_body(name="cube", pos=[cx, cy, H + s + 1e-3])
            cube.add_freejoint(name="cube_joint")
            _box(cube, "cube_geom", [s, s, s], [0, 0, 0], [0.85, 0.1, 0.1, 1], 1, 1, friction=(cfg.cube_friction, 0.02, 0.002),
                 mass=cfg.cube_mass, condim=4, solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])
        if for_planning:                     # cube welded to the hand, offset along tool z (see planner.CARRY_OFFSET)
            s = cfg.cube_size / 2
            off = grip.tcp_z + (cfg.grasp_h - s)           # tool z points DOWN; cube centre is (grasp_h - s) below the TCP
            ee.add_geom(name="carried_cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[s, s, s], pos=[0, 0, off],
                        contype=8, conaffinity=8, rgba=[1, 0.5, 0, 0.0], group=4, mass=1e-6)
    scene = Scene(cfg, spec.compile(), grip, design, for_planning)
    return scene.finalize()


def colliding_geoms_of_bodies(model: mujoco.MjModel, body_names):
    """ids of geoms that take part in collisions (contype/conaffinity != 0) on exactly these bodies"""
    ids = {model.body(n).id for n in body_names}
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] in ids and (model.geom_contype[g] or model.geom_conaffinity[g])]


def pad_geom_ids(model: mujoco.MjModel, grip) -> list:
    """pad geoms = explicitly named pads (attached grippers) + all colliding geoms of the pad bodies (built-in hands)"""
    ids = [model.geom(n).id for n in grip.pad_geoms]
    if grip.pad_bodies: ids += colliding_geoms_of_bodies(model, grip.pad_bodies)
    return sorted(set(ids))


def geom_ids_under(model: mujoco.MjModel, body_id: int):
    """All geom ids whose body is `body_id` or a descendant (used for 'the hand' = gripper + end effector)."""
    out = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        while b != 0 and b != body_id: b = model.body_parentid[b]
        if b == body_id: out.append(g)
    return out
