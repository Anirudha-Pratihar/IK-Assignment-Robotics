"""Robot registry.  ONE place that knows robot-specific names, so the rest of pick_place/ is robot-agnostic.
   heal   : 6-DOF HEAL arm  + attached gripper (Robotiq 2F-85 or built-in fallback)   [the original, unchanged behaviour]
   franka : 7-DOF Franka Panda + its built-in two-finger hand (redundant arm -> null-space posture term)
To add a robot: add one RobotSpec below (names of joints / actuators / bodies, home posture, workspace grid)."""
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DESC = REPO / "robot_descriptions"


@dataclass
class RobotSpec:
    key: str
    label: str
    arm_xml: str
    arm_joints: tuple
    arm_actuators: tuple
    ee_body: str                 # body that carries the TCP site (hand / flange)
    gripper_mode: str            # 'attach' (gripper spec attached at tool_site) | 'builtin' (hand is part of arm_xml)
    tool_site: str
    arm_bodies: tuple            # bodies whose colliding geoms count as "arm" for contact classification
    env_bodies: tuple            # subset used for arm-vs-environment distance constraints
    replace_servo: bool          # True: swap raw torque motors for position servos (HEAL); False: keep the model's own servos
    q_nom: tuple                 # nominal posture (IK seeds, null-space target)
    cfg_overrides: dict
    grid: dict                   # workspace analysis grid / table candidates
    base_xy: tuple
    design_json: str
    out_dir: str
    pad_bodies: tuple = ()       # builtin hands: bodies whose colliding geoms are the pads
    pedestal: tuple = None       # (x, y, z_top) visual column under a raised base
    material_fix: str = None


ROBOTS = {
    "heal": RobotSpec(
        key="heal", label="HEAL 6-DOF",
        arm_xml=str(DESC / "single_arm_heal_effort_actuation_rs_mj.xml"),
        arm_joints=tuple(f"joint_{i}" for i in range(1, 7)),
        arm_actuators=("turret", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"),
        ee_body="end_effector", gripper_mode="attach", tool_site="right_center",
        arm_bodies=("link_1", "link_2", "link_3", "link_4", "link_5", "end_effector"),
        env_bodies=("link_2", "link_3", "link_4", "link_5", "end_effector"),
        replace_servo=True, q_nom=(0.0, -1.0, 1.5, 0.0, 1.2, 0.0),
        cfg_overrides={},
        grid=dict(x=(0.20, 0.80), y=(-0.54, 0.54), H=(0.15, 0.20, 0.25, 0.30), x_min_table=0.22),
        base_xy=(0.0, 0.0), design_json=str(REPO / "configs" / "table_design.json"), out_dir=str(REPO / "results"),
        material_fix="arm"),
    "franka": RobotSpec(
        key="franka", label="Franka Panda 7-DOF",
        arm_xml=str(DESC / "franka" / "scene_fk.xml"),
        arm_joints=tuple(f"joint{i}" for i in range(1, 8)),
        arm_actuators=tuple(f"actuator{i}" for i in range(1, 8)),
        ee_body="hand", gripper_mode="builtin", tool_site="",
        arm_bodies=("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7"),
        env_bodies=("link2", "link3", "link4", "link5", "link6", "link7"),
        replace_servo=False, q_nom=(0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785),
        cfg_overrides=dict(n_arm=7, ee_body="hand", tool_site="", grasp_h=0.025, release_h=0.050, posture_gain=0.5, vmax=2.0, spawn_tray_margin=0.14, pads_tray_band=0.004),
        # base is mounted at (-0.3, 0, 0.8) in panda.xml, so the table is searched at heights just below the base
        grid=dict(x=(-0.10, 0.60), y=(-0.50, 0.50), H=(0.55, 0.60, 0.65, 0.70, 0.75), x_min_table=0.0),
        base_xy=(-0.3, 0.0), design_json=str(REPO / "configs" / "franka" / "table_design.json"), out_dir=str(REPO / "results" / "franka"),
        pad_bodies=("left_finger", "right_finger"), pedestal=(-0.3, 0.0, 0.8)),
}


def get_robot(key: str) -> RobotSpec:
    if key not in ROBOTS: raise KeyError(f"unknown robot '{key}', choose from {list(ROBOTS)}")
    return ROBOTS[key]
