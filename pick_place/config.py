"""All tunable numbers live here (SI units).  Override any field from the CLI with --set key=value
or by passing a JSON file (--config configs/my.json)."""
import json, os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DESC = REPO / "robot_descriptions"


@dataclass
class Config:
    # ---------------- robot / gripper ----------------
    robot: str = "heal"                 # 'heal' | 'franka'  (see pick_place/robots.py)
    posture_gain: float = 0.0           # null-space pull towards the nominal posture [1/s]; 0 for the non-redundant HEAL (unchanged behaviour)
    posture_cost: float = 1e-3
    franka_grip_kp: float = 1000.0      # stiffer finger servo for the Franka hand (stock gain gives ~1 N per finger)
    arm_xml: str = str(DESC / "single_arm_heal_effort_actuation_rs_mj.xml")
    gripper_xml: str = str(DESC / "robotiq_2f85_v4" / "2f85.xml")
    gripper_kind: str = "auto"          # auto -> robotiq if assets/*.stl exist, else 'fallback'; or force 'robotiq'|'fallback'
    n_arm: int = 6
    ee_body: str = "end_effector"       # last arm body; gripper is attached to its 'right_center' site
    tool_site: str = "right_center"
    tcp_z_robotiq: float = 0.130        # pad-centre distance from the flange along tool z (2f85.xml pads: z=0.12..0.139)
    tcp_z_fallback: float = 0.120
    # ---------------- simulation / servo ----------------
    timestep: float = 0.002
    ik_dt: float = 0.02                 # IK / planning tick (50 Hz)
    servo_kp: float = 1500.0            # joint position servo (replaces the raw motors of the HEAL xml)
    servo_kv: float = 80.0
    # ---------------- objects ----------------
    cube_size: float = 0.04             # edge length
    cube_mass: float = 0.05
    cube_friction: float = 1.2
    table_thickness: float = 0.04
    tray_inner: tuple = (0.17, 0.17)    # inner x,y size of the tray
    tray_wall_h: float = 0.05
    tray_wall_t: float = 0.008
    tray_base_t: float = 0.008
    # ---------------- task geometry (heights are TCP height above table top) ----------------
    grasp_h: float = 0.030              # pads centred 3 cm above the table (cube is 4 cm tall)
    pregrasp_h: float = 0.12
    lift_h: float = 0.14                # carry height: cube bottom = lift_h-0.03 must clear the tray walls (0.05) + margin
    release_h: float = 0.055            # tcp height above TABLE top at release (cube bottom ~17 mm above the tray floor)
    # ---------------- planning / IK ----------------
    vmax: float = 2.0                   # joint velocity limit [rad/s] used by QP/mink and for violation counting
    v_cart: float = 0.30                # average Cartesian speed [m/s]
    w_cart: float = 1.2                 # average angular speed [rad/s]
    min_seg_time: float = 0.8
    ik_gain: float = 0.8                # fraction of the task error closed per tick (closed-loop gain)
    dls_lambda: float = 0.03
    dls_clip: bool = True               # DLS has no native limits: clip q to range (position only)
    dls_vel_scale: bool = False         # DLS: scale dq to vmax (False = leave unbounded so violations are measurable)
    qp_damping: float = 1e-3
    solver: str = "daqp"                # qpsolvers backend
    d_min: float = 0.010                # keep-out distance for collision constraints
    d_detect: float = 0.060             # constraints are created for pairs closer than this
    pos_tol: float = 2.0e-3
    rot_tol: float = 0.03
    settle_max: int = 150               # extra ticks allowed to converge at a waypoint
    max_ticks_per_seg: int = 600
    # ---------------- episodes ----------------
    n_episodes: int = 25
    seed: int = 0
    methods: tuple = ("mink", "dls", "qp")
    grasp_yaw_retries: int = 1          # how many alternative grasp yaws to try if IK fails (mitigation); 1 = no retry
    use_collision_avoidance: bool = True   # reactive constraints (QP rows / mink CollisionAvoidanceLimit / DLS repulsion)
    smart_lift: bool = True             # geometry-aware carry height: lift at least to (tray wall + carry_clearance + grasp_h)
    spawn_tray_margin: float = 0.07     # cube centre must be at least this far from the tray's outer wall at spawn (Franka hand is wider -> larger)
    pads_tray_band: float = 0.015       # DLS repulsion band for pads-vs-tray (small for the Franka: fingertips intentionally approach the tray floor)
    carry_clearance: float = 0.030      # cube bottom must pass this far above the tray wall top
    limit_scale: float = 1.0            # <1 shrinks the arm joint ranges about their centre (stress test)
    settle_time_close: float = 0.8
    settle_time_open: float = 0.6
    out_dir: str = str(REPO / "results")
    design_json: str = str(REPO / "configs" / "table_design.json")

    # ---- helpers ----
    def update(self, **kw):
        names = {f.name for f in fields(self)}
        for k, v in kw.items():
            if k not in names: raise KeyError(f"unknown config key '{k}'")
            cur = getattr(self, k)
            if isinstance(cur, bool):  v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            elif isinstance(cur, int): v = int(v)
            elif isinstance(cur, float): v = float(v)
            elif isinstance(cur, tuple) and isinstance(v, str): v = tuple(v.split(","))
            setattr(self, k, v)
        return self

    def apply_robot(self, key):
        from .robots import get_robot
        r = get_robot(key); self.robot = key
        self.arm_xml, self.design_json, self.out_dir = r.arm_xml, r.design_json, r.out_dir
        self.n_arm, self.ee_body, self.tool_site = len(r.arm_joints), r.ee_body, r.tool_site
        self.update(**r.cfg_overrides)
        return self

    @staticmethod
    def load(path=None, overrides=None, robot=None):
        c = Config()
        if robot: c.apply_robot(robot)
        if path: c.update(**json.load(open(path)))
        if overrides: c.update(**dict(s.split("=", 1) for s in overrides))
        return c

    def to_json(self): return json.dumps(asdict(self), indent=2)


def load_design(cfg: Config):
    """Table / tray / spawn-region design produced by scripts/01_workspace_analysis.py."""
    if not os.path.exists(cfg.design_json):
        raise FileNotFoundError(f"{cfg.design_json} not found - run scripts/01_workspace_analysis.py first "
                                f"(or copy configs/table_design.example.json).")
    return json.load(open(cfg.design_json))
