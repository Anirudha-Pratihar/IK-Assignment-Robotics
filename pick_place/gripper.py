"""Gripper factory.  Returns an MjSpec ready for `arm_spec.attach(...)` plus the facts the rest of the code needs.
 * 'robotiq'  : the 2F-85 from robot_descriptions/robotiq_2f85_v4 (needs its assets/*.stl, which were NOT in the zip)
 * 'fallback' : a primitive-geometry parallel-jaw gripper with the same interface (jaws close along tool x,
                one actuator 'fingers_actuator' with ctrl 0 (open) .. 255 (closed), pads named left_pad1/right_pad1).
 Both expose: actuator name, open/close ctrl, pad geom names (after the 'gripper/' prefix), tool-z TCP offset."""
import os
from dataclasses import dataclass
import mujoco

FALLBACK_XML = """
<mujoco model="fallback_gripper">
  <compiler angle="radian"/>
  <default>
    <geom contype="1" conaffinity="1" condim="4" friction="1.6 0.05 0.01" solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <joint damping="2" armature="0.002"/>
  </default>
  <worldbody>
    <body name="base">
      <inertial pos="0 0 0.04" mass="0.7" diaginertia="8e-4 8e-4 4e-4"/>
      <geom name="palm" type="box" size="0.05 0.03 0.012" pos="0 0 0.062" rgba="0.15 0.15 0.15 1"/>
      <geom name="coupling" type="cylinder" size="0.035 0.03" pos="0 0 0.03" rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>
      <body name="left_pad" pos="-0.0433 0 0.120">
        <inertial pos="0 0 0" mass="0.03" diaginertia="1e-5 1e-5 1e-5"/>
        <joint name="left_slide" type="slide" axis="1 0 0" range="0 0.045"/>
        <geom name="left_pad1" type="box" size="0.004 0.011 0.019" rgba="0.5 0.5 0.5 1" mass="1e-3"/>
        <geom name="left_arm" type="box" size="0.004 0.011 0.03" pos="-0.004 0 -0.030" rgba="0.25 0.25 0.25 1" mass="1e-3" contype="0" conaffinity="0"/>
      </body>
      <body name="right_pad" pos="0.0433 0 0.120">
        <inertial pos="0 0 0" mass="0.03" diaginertia="1e-5 1e-5 1e-5"/>
        <joint name="right_slide" type="slide" axis="-1 0 0" range="0 0.045"/>
        <geom name="right_pad1" type="box" size="0.004 0.011 0.019" rgba="0.5 0.5 0.5 1" mass="1e-3"/>
        <geom name="right_arm" type="box" size="0.004 0.011 0.03" pos="0.004 0 -0.030" rgba="0.25 0.25 0.25 1" mass="1e-3" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <equality><joint joint1="left_slide" joint2="right_slide" polycoef="0 1 0 0 0"/></equality>
  <tendon><fixed name="split"><joint joint="left_slide" coef="0.5"/><joint joint="right_slide" coef="0.5"/></fixed></tendon>
  <actuator>
    <!-- ctrl 0..255 -> target finger travel 0..0.045 m (same convention as the Robotiq xml) -->
    <general name="fingers_actuator" tendon="split" ctrlrange="0 255" forcerange="-60 60"
             gainprm="0.7058824 0 0" biasprm="0 -400 -20" biastype="affine"/>
  </actuator>
</mujoco>
"""


@dataclass
class GripperInfo:
    kind: str
    spec: "mujoco.MjSpec"
    actuator: str            # name AFTER attach (prefix applied)
    open_cmd: float
    close_cmd: float
    tcp_z: float
    pad_geoms: tuple         # names after attach
    prefix: str = "gripper/"
    pad_bodies: tuple = ()   # builtin hands: pads = colliding geoms of these bodies


def robotiq_available(cfg) -> bool:
    return os.path.isfile(os.path.join(os.path.dirname(cfg.gripper_xml), "assets", "base.stl"))


def make_gripper(cfg) -> GripperInfo:
    from .robots import get_robot
    if get_robot(cfg.robot).gripper_mode == "builtin":     # Franka hand: ctrl 255 = OPEN, 0 = closed (opposite of the Robotiq xml)
        return GripperInfo("franka_hand", None, "actuator8", 255.0, 0.0, 0.103, (), prefix="", pad_bodies=("left_finger", "right_finger"))
    kind = cfg.gripper_kind
    if kind == "auto": kind = "robotiq" if robotiq_available(cfg) else "fallback"
    if kind == "robotiq":
        if not robotiq_available(cfg):
            raise FileNotFoundError("Robotiq assets missing: put the 2F-85 meshes in robot_descriptions/robotiq_2f85_v4/assets/")
        spec = mujoco.MjSpec.from_file(cfg.gripper_xml)
        return GripperInfo("robotiq", spec, "gripper/fingers_actuator", 0.0, 255.0, cfg.tcp_z_robotiq,
                           ("gripper/left_pad1", "gripper/left_pad2", "gripper/right_pad1", "gripper/right_pad2"))
    spec = mujoco.MjSpec.from_string(FALLBACK_XML)
    return GripperInfo("fallback", spec, "gripper/fingers_actuator", 0.0, 255.0, cfg.tcp_z_fallback,
                       ("gripper/left_pad1", "gripper/right_pad1"))
