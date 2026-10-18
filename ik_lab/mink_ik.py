"""CASE 3 - mink (https://github.com/kevinzakka/mink): differential IK as a QP, built on MuJoCo.
   Tasks + limits are declarative; note mink task gain is a per-step FRACTION of the error (1.0 = close it in one dt), unlike K in CASE 2. mink assembles and solves the same QP as CASE 2 for us."""
import numpy as np
import mujoco, mink
from .common import IKResult, EE_SITE, TOL_P, TOL_R
from scipy.spatial.transform import Rotation as Rot


def solve_mink(prob, q0, p_t, R_t, max_iter=500, dt=0.02, solver="daqp", damping=1e-3,
               pos_cost=1.0, ori_cost=1.0, gain=0.5, vmax=10.0, record=False, pos_only=False):
    cfg = mink.Configuration(prob.model)
    q_full = np.zeros(prob.model.nq); q_full[prob.qadr] = np.clip(q0, prob.qmin, prob.qmax)
    cfg.update(q_full)
    task = mink.FrameTask(EE_SITE, "site", position_cost=pos_cost, orientation_cost=0.0 if pos_only else ori_cost,
                          lm_damping=0.0)
    task.gain = gain
    T = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R_t), p_t)
    task.set_target(T)
    limits = [mink.ConfigurationLimit(prob.model),
              mink.VelocityLimit(prob.model, {prob.model.joint(i).name: vmax for i in prob.jids})]
    traj = []
    for it in range(max_iter):
        err = task.compute_error(cfg)
        ep, er = np.linalg.norm(err[:3]), np.linalg.norm(err[3:])
        if ep < TOL_P and (pos_only or er < TOL_R):
            return IKResult(cfg.q[prob.qadr].copy(), True, it, ep, 0 if pos_only else er, traj, "mink")
        v = mink.solve_ik(cfg, [task], dt, solver, damping=damping, limits=limits)
        cfg.integrate_inplace(v, dt)
        if record: traj.append(cfg.q[prob.qadr].copy())
    err = task.compute_error(cfg)
    return IKResult(cfg.q[prob.qadr].copy(), False, max_iter, np.linalg.norm(err[:3]), np.linalg.norm(err[3:]), traj, "mink")
