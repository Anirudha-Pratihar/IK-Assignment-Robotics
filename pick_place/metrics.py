"""Per-episode log record, failure taxonomy, aggregation helpers."""
from dataclasses import dataclass, field, asdict
import json
import numpy as np

FAIL = ["ik_infeasible", "ik_limits", "collision", "grasp_miss", "grasp_slip", "place_miss", "sim_unstable", "other"]


@dataclass
class EpisodeLog:
    episode: int = 0
    seed: int = 0
    method: str = ""
    cube_x: float = 0.0
    cube_y: float = 0.0
    cube_yaw: float = 0.0
    success: bool = False
    failure_reason: str = ""
    failure_phase: str = ""
    failure_detail: str = ""
    phase_reached: str = ""
    grasp_phi: float = 0.0
    yaw_attempts: int = 1
    plan_time_s: float = 0.0           # wall time spent inside IK steppers (time to solve)
    ik_iters: int = 0                  # sum over waypoints of IK iterations to converge (fixed target, from the segment start)
    track_ticks: int = 0               # IK ticks spent tracking the Cartesian reference (deterministic from segment timing)
    conv_fail: int = 0                 # waypoints where the one-shot converge solve did not reach tolerance
    ik_iters_by_phase: dict = field(default_factory=dict)
    min_clear_plan: float = float("nan")   # min signed distance predicted by the plan (robot/hand vs table+tray, carried cube vs tray)
    min_clear_robot_env: float = float("nan")  # measured in sim: arm+hand (no pads) vs table/tray
    min_clear_cube_tray: float = float("nan")  # measured in sim while carrying
    limit_events: int = 0              # ticks where the raw update left the joint range (DLS: clipped)
    vel_events: int = 0                # ticks where |dq| > vmax
    qp_fail: int = 0
    max_joint_vel: float = 0.0
    max_seg_pos_err_mm: float = 0.0
    track_rms_mm: float = float("nan")     # executed TCP vs planned TCP
    exec_time_s: float = 0.0           # simulated time
    wall_time_s: float = 0.0
    place_err_mm: float = float("nan")     # cube centre vs tray centre (xy)
    cube_final: list = field(default_factory=list)

    def to_dict(self): return asdict(self)


def save_jsonl(logs, path):
    with open(path, "w") as f:
        for l in logs: f.write(json.dumps(l.to_dict() if hasattr(l, "to_dict") else l) + "\n")


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]
