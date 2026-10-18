"""Three differential-IK 'steppers' with ONE interface:   dq = stepper.step(q, p_target, R_target)
   All close the same fraction (cfg.ik_gain) of the 6-D pose error per tick so the comparison is about the solver, not the gain.
     DLSStepper   closed-loop damped least squares  (+ potential-field repulsion, position clipping only)
     QPStepper    closed-loop QP: joint position + velocity limits and collision-distance inequalities (qpsolvers / daqp)
     MinkStepper  the off-the-shelf baseline: mink FrameTask + ConfigurationLimit + VelocityLimit + CollisionAvoidanceLimit
"""
import time
import numpy as np
import mujoco
import qpsolvers
from ik_lab.common import pose_error6
from .collision import qp_rows, repulsion_dq
from .robots import get_robot


class StepStats:
    def __init__(self): self.reset()
    def reset(self):
        self.ticks = 0; self.solve_time = 0.0; self.qp_fail = 0; self.limit_events = 0; self.vel_events = 0
        self.active_constraints = 0; self.max_dq = 0.0


class BaseStepper:
    name = "base"
    def __init__(self, cfg, prob, csets):
        self.cfg, self.prob, self.csets = cfg, prob, csets
        self.stats = StepStats(); self.groups = []; self.dt = cfg.ik_dt
        self.n = prob.n; self.q_nom = np.array(get_robot(cfg.robot).q_nom, float)

    def posture_dq(self, q):
        """desired joint velocity towards the nominal posture (used only in the null space of redundant arms)"""
        return self.cfg.posture_gain * (self.q_nom - q)

    def begin_segment(self, phase):
        self.groups = self.csets.groups(phase, self.cfg.use_collision_avoidance)

    def task_velocity(self, q, p_t, R_t):
        p, R = self.prob.fk(q)
        e = pose_error6(p, R, p_t, R_t)
        return self.cfg.ik_gain * e / self.dt, e

    def step(self, q, p_t, R_t):
        t0 = time.perf_counter()
        dq = self._solve(q, p_t, R_t)
        self.stats.solve_time += time.perf_counter() - t0
        self.stats.ticks += 1; self.stats.max_dq = max(self.stats.max_dq, float(np.abs(dq).max()))
        return dq

    def integrate(self, q, dq):
        """q_next.  Returns (q_next, limit_event, vel_event) - DLS clips, the others should never need to"""
        qn = q + dq * self.dt
        lim = bool(np.any(qn < self.prob.qmin - 1e-6) or np.any(qn > self.prob.qmax + 1e-6))
        vel = bool(np.abs(dq).max() > self.cfg.vmax * 1.001)
        self.stats.limit_events += lim; self.stats.vel_events += vel
        return qn, lim, vel


class DLSStepper(BaseStepper):
    name = "dls"
    def _solve(self, q, p_t, R_t):
        cfg, prob = self.cfg, self.prob
        v, _ = self.task_velocity(q, p_t, R_t)
        J = prob.jac(q)
        lam2 = cfg.dls_lambda ** 2
        smin = np.linalg.svd(J, compute_uv=False)[-1]                 # adaptive damping near singularities
        if smin < 0.03: lam2 += (1 - (smin / 0.03) ** 2) * 0.15 ** 2
        A = np.linalg.solve(J @ J.T + lam2 * np.eye(6), np.hstack([v[:, None], J]))
        dq = J.T @ A[:, 0]
        if cfg.posture_gain > 0:                                       # redundant arm: project posture pull into the null space
            dq = dq + (np.eye(self.n) - J.T @ A[:, 1:]) @ self.posture_dq(q)
        if self.groups:
            rep, n = repulsion_dq(prob, self.groups, cfg.d_detect); dq = dq + rep; self.stats.active_constraints += n
        if cfg.dls_vel_scale:
            s = np.abs(dq).max()
            if s > cfg.vmax: dq *= cfg.vmax / s
        return dq

    def integrate(self, q, dq):
        qn, lim, vel = super().integrate(q, dq)
        if self.cfg.dls_clip: qn = np.clip(qn, self.prob.qmin, self.prob.qmax)
        return qn, lim, vel


class QPStepper(BaseStepper):
    name = "qp"
    def _solve(self, q, p_t, R_t):
        cfg, prob = self.cfg, self.prob
        v, _ = self.task_velocity(q, p_t, R_t)
        J = prob.jac(q)
        H = J.T @ J + cfg.qp_damping * np.eye(self.n); g = -J.T @ v
        if cfg.posture_gain > 0: g = g - cfg.qp_damping * self.posture_dq(q)   # null-space posture (cost mu/2 |dq - dq_posture|^2)
        lb = np.maximum(-cfg.vmax, (prob.qmin - q) / self.dt); ub = np.minimum(cfg.vmax, (prob.qmax - q) / self.dt)
        G, h = (None, None)
        if self.groups:
            G, h = qp_rows(prob, self.groups, self.dt, cfg.d_detect)
            if G is not None: self.stats.active_constraints += len(h)
        dq = qpsolvers.solve_qp(H, g, G=G, h=h, lb=lb, ub=ub, solver=cfg.solver)
        if dq is None and G is not None:                              # infeasible collision rows -> drop them, keep limits
            self.stats.qp_fail += 1
            dq = qpsolvers.solve_qp(H, g, lb=lb, ub=ub, solver=cfg.solver)
        if dq is None: self.stats.qp_fail += 1; dq = np.zeros(self.n)
        return dq


class MinkStepper(BaseStepper):
    name = "mink"
    def __init__(self, cfg, prob, csets):
        super().__init__(cfg, prob, csets)
        import mink
        self.mink = mink; m = prob.model
        self.conf = mink.Configuration(m)
        self.task = mink.FrameTask("tcp", "site", position_cost=1.0, orientation_cost=1.0, lm_damping=0.0)
        self.task.gain = cfg.ik_gain
        self.base_limits = [mink.ConfigurationLimit(m),
                            mink.VelocityLimit(m, {m.joint(i).name: cfg.vmax for i in range(cfg.n_arm)})]
        self.limits = list(self.base_limits)
        self.ptask = None
        if cfg.posture_gain > 0:                                       # null-space posture task (tiny cost -> only acts in the null space)
            self.ptask = mink.PostureTask(m, cost=cfg.posture_cost); self.ptask.gain = min(1.0, cfg.posture_gain * cfg.ik_dt)

    def begin_segment(self, phase):
        super().begin_segment(phase)
        mk, m, cfg = self.mink, self.prob.model, self.cfg
        self.limits = list(self.base_limits)
        for gr in self.groups:      # one CollisionAvoidanceLimit per group so each keeps its own d_min
            self.limits.append(mk.CollisionAvoidanceLimit(m, [(gr.g1, gr.g2)], gain=0.85, minimum_distance_from_collisions=gr.d_min,
                                                          collision_detection_distance=cfg.d_detect))

    def _solve(self, q, p_t, R_t):
        mk = self.mink
        qf = self.prob.data.qpos.copy(); qf[self.prob.qadr] = q      # arm from planner, cube / gripper from the synced state
        self.conf.update(qf)
        self.task.set_target(mk.SE3.from_rotation_and_translation(mk.SO3.from_matrix(R_t), p_t))
        try:
            tasks = [self.task]
            if self.ptask is not None:
                qt = qf.copy(); qt[self.prob.qadr] = self.q_nom; self.ptask.set_target(qt); tasks.append(self.ptask)
            v = mk.solve_ik(self.conf, tasks, self.dt, self.cfg.solver, damping=self.cfg.qp_damping, limits=self.limits)
        except mk.exceptions.NoSolutionFound:
            self.stats.qp_fail += 1; return np.zeros(self.n)
        return v[self.prob.dofs]


def make_stepper(method, cfg, prob, csets):
    return {"dls": DLSStepper, "qp": QPStepper, "mink": MinkStepper}[method](cfg, prob, csets)
