"""CASE 1 - Damped Least Squares (Levenberg-Marquardt) differential IK.
   dq = J^T (J J^T + lam^2 I)^-1 e          (lam = damping; adaptive variant below)
   Open-loop in the sense that we integrate dq onto q at each step; the 'closed-loop' version
   re-evaluates the pose error from FK every iteration, which is what this loop already does.
   Limits are handled only by clipping (no guarantee)."""
import numpy as np
from .common import IKResult, pose_error6, TOL_P, TOL_R


def solve_dls(prob, q0, p_t, R_t, max_iter=500, lam=0.01, step=1.0, adaptive=True,
              pos_only=False, null_posture=None, k_null=0.0, record=False):
    q = prob.clip(np.asarray(q0, float)); traj = []
    for it in range(max_iter):
        p, R = prob.fk(q)
        e = pose_error6(p, R, p_t, R_t)
        if pos_only: e = e[:3]
        if np.linalg.norm(e[:3]) < TOL_P and (pos_only or np.linalg.norm(e[3:]) < TOL_R):
            return IKResult(q, True, it, np.linalg.norm(e[:3]), 0 if pos_only else np.linalg.norm(e[3:]), traj, "DLS")
        J = prob.jac(q); J = J[:3] if pos_only else J
        # adaptive damping (Nakamura-Hanafusa): lam^2 ramps up only when sigma_min < eps
        lam2 = lam ** 2
        if adaptive:
            smin, eps, lam_max = np.linalg.svd(J, compute_uv=False)[-1], 0.03, 0.15
            if smin < eps: lam2 = lam ** 2 + (1 - (smin / eps) ** 2) * lam_max ** 2
        JJt = J @ J.T + lam2 * np.eye(J.shape[0])
        dq = J.T @ np.linalg.solve(JJt, e)
        if null_posture is not None and k_null > 0:            # secondary task in the null space
            Jp = J.T @ np.linalg.solve(JJt, J)
            dq += (np.eye(prob.n) - Jp) @ (k_null * (null_posture - q))
        q = prob.clip(q + step * dq)
        if record: traj.append(q.copy())
    p, R = prob.fk(q); e = pose_error6(p, R, p_t, R_t)
    return IKResult(q, False, max_iter, np.linalg.norm(e[:3]), np.linalg.norm(e[3:]), traj, "DLS")
