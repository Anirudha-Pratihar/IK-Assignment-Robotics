"""CASE 2 - Closed-loop Quadratic-Programming IK with joint position + velocity limits.
   Each control tick solve   min_dq  1/2 || J dq - K e ||^2_W + 1/2 lam ||dq||^2
                              s.t.   (qmin - q)/dt*alpha <= dq <= (qmax - q)/dt*alpha   (position limits)
                                     -vmax <= dq <= vmax                                  (velocity limits)
   'Closed-loop' = the task error e is recomputed from the CURRENT (measured) pose every tick and
   fed back with gain K, so tracking error is regulated, not just integrated."""
import numpy as np
import qpsolvers
from .common import IKResult, pose_error6, TOL_P, TOL_R


def qp_step(prob, q, e, dt, K=25.0, lam=1e-3, vmax=10.0, alpha=1.0, w_pos=1.0, w_rot=1.0,
            pos_only=False, solver="daqp", dq_prev=None):
    J = prob.jac(q); W = np.diag([w_pos] * 3 + [w_rot] * 3)
    if pos_only: J, e, W = J[:3], e[:3], W[:3, :3]
    H = J.T @ W @ J + lam * np.eye(prob.n)
    g = -J.T @ W @ (K * e)
    lo = np.maximum(-vmax * np.ones(prob.n), alpha * (prob.qmin - q) / dt)
    hi = np.minimum(+vmax * np.ones(prob.n), alpha * (prob.qmax - q) / dt)
    dq = qpsolvers.solve_qp(H, g, lb=lo, ub=hi, solver=solver)
    return np.zeros(prob.n) if dq is None else dq


def solve_qp(prob, q0, p_t, R_t, max_iter=500, dt=0.02, K=25.0, lam=1e-3, vmax=10.0,
             pos_only=False, record=False, solver="daqp"):
    q = prob.clip(np.asarray(q0, float)); traj = []
    for it in range(max_iter):
        p, R = prob.fk(q); e = pose_error6(p, R, p_t, R_t)
        ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
        if ep < TOL_P and (pos_only or er < TOL_R):
            return IKResult(q, True, it, ep, 0 if pos_only else er, traj, "QP")
        dq = qp_step(prob, q, e, dt, K, lam, vmax, pos_only=pos_only, solver=solver)
        q = q + dq * dt                       # limits are in the QP -> no clipping needed
        if record: traj.append(q.copy())
    p, R = prob.fk(q); e = pose_error6(p, R, p_t, R_t)
    return IKResult(q, False, max_iter, np.linalg.norm(e[:3]), np.linalg.norm(e[3:]), traj, "QP")
