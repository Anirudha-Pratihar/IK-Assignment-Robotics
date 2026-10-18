"""Cartesian segment planner.  A segment = straight-line TCP motion (min-jerk timing) tracked by a closed-loop IK stepper on a
PRIVATE copy of the model (plan model).  Output: joint trajectory Q (n,6) + velocities dQ, plus per-segment metrics."""
from dataclasses import dataclass, field
import numpy as np
import mujoco
from ik_lab import IKProblem
from ik_lab.common import pose_error6
from .geometry import min_jerk, interp_pose, segment_time
from .collision import CollisionSets, min_clearance
from .ik_solvers import make_stepper


@dataclass
class SegResult:
    phase: str
    Q: np.ndarray
    dQ: np.ndarray
    ok: bool
    ticks: int = 0
    solve_time: float = 0.0
    pos_err: float = 0.0
    rot_err: float = 0.0
    min_clear: float = 9.9
    limit_events: int = 0
    vel_events: int = 0
    qp_fail: int = 0
    max_dq: float = 0.0
    reason: str = ""
    conv_iters: int = 0
    conv_time: float = 0.0
    conv_ok: bool = True


class Planner:
    def __init__(self, cfg, plan_scene, method):
        self.cfg, self.sc = cfg, plan_scene
        self.prob = IKProblem(model=plan_scene.model, ee_site="tcp", n_arm=cfg.n_arm)
        self.csets = CollisionSets(plan_scene, cfg)
        self.stepper = make_stepper(method, cfg, self.prob, self.csets)
        self.method = method

    def sync(self, qpos_full):
        """copy the simulated state (arm, gripper, cube) into the plan model"""
        self.prob.data.qpos[:] = qpos_full
        mujoco.mj_kinematics(self.prob.model, self.prob.data)

    def tcp(self, q): return self.prob.fk(q)

    def converge(self, q0, p1, R1, phase, max_iter=300):
        """One-shot pose IK to the waypoint (fixed target, no reference trajectory): iterations + wall time to tolerance."""
        cfg, prob, st = self.cfg, self.prob, self.stepper
        st.stats.reset(); st.begin_segment(phase); q = np.array(q0, float); ok = False
        for it in range(max_iter):
            p, R = prob.fk(q); e = pose_error6(p, R, p1, R1)
            if np.linalg.norm(e[:3]) < cfg.pos_tol * 0.5 and np.linalg.norm(e[3:]) < cfg.rot_tol * 0.5: ok = True; break
            dq = st.step(q, p1, R1); q, _, _ = st.integrate(q, dq)
        return it, st.stats.solve_time, ok

    def plan_segment(self, q0, p1, R1, phase, v_scale=1.0):
        cfg, prob, st = self.cfg, self.prob, self.stepper
        dt = cfg.ik_dt
        c_it, c_t, c_ok = self.converge(q0, p1, R1, phase)
        st.stats.reset(); st.begin_segment(phase)
        q = np.array(q0, float); p0, R0 = prob.fk(q)
        T = segment_time(p0, R0, p1, R1, cfg.v_cart * v_scale, cfg.w_cart * v_scale, cfg.min_seg_time)
        n_ref = int(np.ceil(T / dt))
        Q, dQ = [], []
        min_clear = 9.9

        def tick(p_t, R_t):
            nonlocal q, min_clear
            dq = st.step(q, p_t, R_t)
            q_new, _, _ = st.integrate(q, dq)
            dQ.append((q_new - q) / dt if self.method == "dls" and cfg.dls_clip else dq); Q.append(q_new.copy()); q = q_new
            if self.stepper.groups:
                prob.fk(q); min_clear = min(min_clear, min_clearance(prob.model, prob.data, self.stepper.groups, 0.15))

        for k in range(1, n_ref + 1):
            s, _ = min_jerk(k / n_ref)
            p_t, R_t = interp_pose(p0, R0, p1, R1, s)
            tick(p_t, R_t)
        ok, extra = False, 0
        while True:
            p, R = prob.fk(q); e = pose_error6(p, R, p1, R1)
            pe, re = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            if pe < cfg.pos_tol and re < cfg.rot_tol: ok = True; break
            if extra >= cfg.settle_max or len(Q) >= cfg.max_ticks_per_seg: break
            tick(p1, R1); extra += 1
        res = SegResult(phase, np.array(Q), np.array(dQ), ok, ticks=len(Q), solve_time=st.stats.solve_time, pos_err=pe, rot_err=re,
                        min_clear=min_clear, limit_events=st.stats.limit_events, vel_events=st.stats.vel_events,
                        qp_fail=st.stats.qp_fail, max_dq=st.stats.max_dq, conv_iters=c_it, conv_time=c_t, conv_ok=c_ok)
        if not ok:      # label by physical state: a joint sitting on its bound => limits, otherwise unreachable / singular
            at_lim = bool(np.any(np.abs(q - prob.qmin) < 3e-3) or np.any(np.abs(q - prob.qmax) < 3e-3))
            res.reason = "ik_limits" if (at_lim or st.stats.limit_events > 0) else "ik_infeasible"
        return res
