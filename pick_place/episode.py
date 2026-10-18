"""One pick-and-place episode for one IK method.

  home -> approach (above cube) -> descend -> close -> lift -> transit (above tray) -> place -> open -> retreat -> verify
Planning:   approach+descend are DRY-RUN planned first (so alternative grasp yaws can be tried); later segments are planned from the
            measured state right before execution.
Execution:  planned joint trajectories are replayed by the position servo in MuJoCo; every physics step the contacts are classified.
"""
import time
import numpy as np
from .config import Config
from .geometry import topdown_R, tool_phi, wrap
from .metrics import EpisodeLog
from .collision import Group, min_clearance


def sample_cube(design, seed, cfg):
    rng = np.random.default_rng(seed)
    sp = design["spawn"]
    x = rng.uniform(sp["xmin"], sp["xmax"]); y = rng.uniform(sp["ymin"], sp["ymax"]); yaw = rng.uniform(-np.pi, np.pi)
    return float(x), float(y), float(yaw)


class Abort(Exception):
    def __init__(self, reason, phase, detail=""): self.reason, self.phase, self.detail = reason, phase, detail


def run_episode(env, planner, cfg: Config, design: dict, ep: int, seed: int, method: str, recorder=None, shot_dir=None, trace_dir=None):
    t_wall = time.time()
    H, tcp_cx = design["H"], None
    log = EpisodeLog(episode=ep, seed=seed, method=method)
    cx, cy, yaw = sample_cube(design, seed, cfg)
    log.cube_x, log.cube_y, log.cube_yaw = cx, cy, yaw
    q_home = np.array(design["q_home"]); tx, ty = design["tray_center"]
    env.reset(q_home, (cx, cy), yaw)
    env.recorder = recorder
    planner.sync(env.data.qpos)
    st = planner.stepper
    s = cfg.cube_size / 2

    # ---- measurement groups on the SIM model (clearance metrics) ----
    # NOTE: geom ids differ between plan and sim model (plan adds 'carried_cube'), so build groups from the SIM model's own id sets
    m_sim = env.model
    g_robot = [Group("r", sorted(env.arm_env_g | env.hand_g), sorted(env.table_g | env.tray_g), 0)]
    g_cube = [Group("c", [env.id_cube], sorted(env.tray_g), 0)]
    clr_robot, clr_cube = [9.9], [9.9]
    state = dict(phase="init", carrying=False, fail=None)
    tracks, trace = [], []

    def monitor():
        v = env.contact_violations(state["phase"])
        if v:
            k, a, b = v[0]; raise Abort("collision", state["phase"], f"{k}:{env.name_pair(a, b)}")
        if state["carrying"]:
            p, _ = env.tcp()
            if np.linalg.norm(env.cube_pos() - p) > 0.045 or env.cube_pos()[2] < H + 0.5 * s:
                raise Abort("grasp_slip", state["phase"], "cube left the gripper")
        if not np.isfinite(env.data.qpos).all() or np.abs(env.data.qvel).max() > 1e3:
            raise Abort("sim_unstable", state["phase"], "NaN/huge velocity")
        return False

    def tick_cb(k):
        p, _ = env.tcp()
        trace.append([env.data.time, *env.q(), *p, *env.cube_pos(), PHASES.index(state["phase"]) if state["phase"] in PHASES else -1])
        clr_robot.append(env.robot_env_clearance(g_robot) if True else 9.9)
        if state["carrying"]: clr_cube.append(env.robot_env_clearance(g_cube))

    PHASES = ["approach", "descend", "close", "lift", "transit", "place", "release", "retreat", "settle"]

    def account(res):
        log.plan_time_s += res.solve_time + res.conv_time; log.ik_iters += res.conv_iters; log.track_ticks += res.ticks
        log.ik_iters_by_phase[res.phase] = log.ik_iters_by_phase.get(res.phase, 0) + res.conv_iters
        log.conv_fail += (not res.conv_ok)
        log.limit_events += res.limit_events; log.vel_events += res.vel_events; log.qp_fail += res.qp_fail
        log.max_joint_vel = max(log.max_joint_vel, res.max_dq); log.max_seg_pos_err_mm = max(log.max_seg_pos_err_mm, res.pos_err * 1e3)
        log.min_clear_plan = res.min_clear if np.isnan(log.min_clear_plan) else min(log.min_clear_plan, res.min_clear)

    def set_phase(ph):
        state["phase"] = ph
        if recorder is not None: recorder.label = f"{method.upper()} ep{ep} {ph}"

    def execute(res):
        set_phase(res.phase)
        env.run_trajectory(res.Q, res.dQ, monitor=monitor, tick_cb=tick_cb)
        # planned vs executed TCP tracking error at the end of the segment
        p_exec, _ = env.tcp(); p_plan, _ = planner.tcp(res.Q[-1]); tracks.append(np.linalg.norm(p_exec - p_plan))

    def plan(q0, p, R, phase):
        planner.sync(env.data.qpos)
        res = planner.plan_segment(q0, p, R, phase); account(res)
        if not res.ok: raise Abort(res.reason, phase, f"pos_err={res.pos_err*1e3:.1f}mm rot_err={res.rot_err:.3f}rad")
        return res

    try:
        # -------- choose grasp yaw: cube yaw + k*pi/2, nearest to the current tool yaw first --------
        _, R0 = env.tcp(); phi0 = tool_phi(R0)
        cands = sorted([wrap(yaw + k * np.pi / 2) for k in range(4)], key=lambda a: abs(wrap(a - phi0)))
        cands = cands[:max(1, cfg.grasp_yaw_retries)]
        lift_h = max(cfg.lift_h, cfg.tray_wall_h + cfg.carry_clearance + cfg.grasp_h) if cfg.smart_lift else cfg.lift_h   # 'smart' carry height
        p_pre = np.array([cx, cy, H + cfg.pregrasp_h]); p_grasp = np.array([cx, cy, H + cfg.grasp_h])
        r_app = r_des = None; last_err = None
        for a, phi in enumerate(cands):
            log.yaw_attempts = a + 1
            try:
                R_g = topdown_R(phi)
                planner.sync(env.data.qpos)
                r_app = planner.plan_segment(env.q(), p_pre, R_g, "approach"); account(r_app)
                if not r_app.ok: raise Abort(r_app.reason, "approach", f"pos_err={r_app.pos_err*1e3:.1f}mm")
                r_des = planner.plan_segment(r_app.Q[-1], p_grasp, R_g, "descend"); account(r_des)
                if not r_des.ok: raise Abort(r_des.reason, "descend", f"pos_err={r_des.pos_err*1e3:.1f}mm")
                log.grasp_phi = float(phi); last_err = None; break
            except Abort as e:
                last_err = e; r_app = r_des = None
        if last_err is not None: raise last_err
        R_g = topdown_R(log.grasp_phi)

        # -------- approach + descend --------
        env.set_gripper(False)
        execute(r_app); execute(r_des)

        # -------- grasp --------
        set_phase("close"); env.set_gripper(True); env.hold(cfg.settle_time_close, monitor)
        if env.cube_pad_contacts() < 2: raise Abort("grasp_miss", "close", f"pad contacts={env.cube_pad_contacts()}")
        state["carrying"] = True

        # -------- lift, transit, place --------
        r = plan(env.q(), np.array([cx, cy, H + lift_h]), R_g, "lift"); execute(r)
        r = plan(env.q(), np.array([tx, ty, H + lift_h]), R_g, "transit"); execute(r)
        r = plan(env.q(), np.array([tx, ty, H + cfg.release_h]), R_g, "place"); execute(r)

        # -------- release + retreat --------
        set_phase("release"); state["carrying"] = False
        env.set_gripper(False); env.hold(cfg.settle_time_open, monitor)
        r = plan(env.q(), np.array([tx, ty, H + lift_h]), R_g, "retreat"); execute(r)
        set_phase("settle"); env.hold(0.6, monitor)

        # -------- verify --------
        cp = env.cube_pos(); ix, iy = cfg.tray_inner
        inside = abs(cp[0] - tx) < ix / 2 - 0.5 * s and abs(cp[1] - ty) < iy / 2 - 0.5 * s and H < cp[2] < H + cfg.tray_wall_h
        log.place_err_mm = float(np.linalg.norm(cp[:2] - np.array([tx, ty])) * 1e3); log.cube_final = [float(v) for v in cp]
        if inside and env.cube_vel() < 0.05: log.success = True
        else: raise Abort("place_miss", "settle", f"cube at {np.round(cp, 3).tolist()}")
        log.phase_reached = "done"
    except Abort as e:
        log.success = False; log.failure_reason, log.failure_phase, log.failure_detail = e.reason, e.phase, e.detail
        log.phase_reached = state["phase"]
        if recorder is not None and shot_dir:
            try: recorder.save_png(f"{shot_dir}/{method}_ep{ep:03d}_{e.reason}_{e.phase}.png", f"{method} ep{ep} FAIL {e.reason} in {e.phase}")
            except Exception as ex: log.failure_detail += f" (screenshot failed: {ex})"
    finally:
        env.recorder = None
    log.min_clear_robot_env = float(min(clr_robot)) if len(clr_robot) > 1 else float('nan'); log.min_clear_cube_tray = float(min(clr_cube)) if len(clr_cube) > 1 else float("nan")
    log.track_rms_mm = float(np.sqrt(np.mean(np.square(tracks))) * 1e3) if tracks else float("nan")
    log.exec_time_s = float(env.data.time); log.wall_time_s = time.time() - t_wall
    if trace_dir and trace:
        import os; os.makedirs(trace_dir, exist_ok=True)
        np.savez_compressed(f"{trace_dir}/{method}_ep{ep:03d}.npz", trace=np.array(trace), cols="t,q1..q6,tcp_xyz,cube_xyz,phase")
    return log
