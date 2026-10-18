#!/usr/bin/env python3
"""Closed-loop Cartesian tracking in MuJoCo using the starter JointVelocityController.
   IK (DLS | QP | mink) turns the Cartesian error measured from the SIMULATED arm into a joint-velocity
   command; the D-controller + gravity compensation from utils/ tracks that velocity.

   python scripts/ik_demo_viewer.py --method qp --view      # watch it
   python scripts/ik_demo_viewer.py --method all            # headless comparison table
"""
import os, sys, argparse
import numpy as np, mujoco, mink
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ik_lab import IKProblem
from ik_lab.common import pose_error6, EE_SITE
from ik_lab.qp import qp_step
from utils.mj_velocity_control.mj_velocity_ctrl import JointVelocityController


class DampingAwareVelocityController(JointVelocityController):
    """Starter controller + feedforward of the joints' passive damping.
    Without it the arm only reaches kd/(kd+damping) of the commanded velocity (HEAL damping = 55 N.m.s/rad)."""
    def control_callback(self, model, data):
        tau = super().control_callback(model, data)
        for i in range(self.num_actuators):
            if i in self.gripper_ids: continue
            ff = model.dof_damping[self.dof_indices[i]] * self.v_targets[i]
            data.ctrl[i] += ff; tau[i] += ff
        return tau


def dls_step(prob, q, e, dt, K=5.0, lam=0.05, vmax=1.0):
    J = prob.jac(q)
    dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), K * e)
    s = np.abs(dq).max()
    return dq * (vmax / s) if s > vmax else dq                    # scale (not clip) to keep direction


class MinkStep:
    def __init__(self, prob, vmax):
        self.cfg = mink.Configuration(prob.model)
        self.task = mink.FrameTask(EE_SITE, "site", position_cost=1.0, orientation_cost=1.0)
        self.lims = [mink.ConfigurationLimit(prob.model),
                     mink.VelocityLimit(prob.model, {prob.model.joint(i).name: vmax for i in prob.jids})]
    def __call__(self, prob, q, p_t, R_t, dt, gain):
        self.cfg.update(np.r_[q, np.zeros(prob.model.nq - prob.n)][:prob.model.nq])
        self.task.gain = gain
        self.task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R_t), p_t))
        return mink.solve_ik(self.cfg, [self.task], dt, "daqp", damping=1e-3, limits=self.lims)[:prob.n]


def run(method, view=False, T=8.0, vmax=1.0, q_start=None):
    prob = IKProblem(); model, data = prob.model, mujoco.MjData(prob.model)
    q0 = np.array([0.14, -1.52, 0.62, 0.69, 1.03, 0.49]) if q_start is None else q_start
    data.qpos[prob.qadr] = q0; mujoco.mj_forward(model, data)
    p_c, R_t = prob.fk(q0)
    ctrl = DampingAwareVelocityController(model, data, kd=20.0); mujoco.set_mjcb_control(ctrl.control_callback)
    ik_dt = 0.02; mstep = MinkStep(prob, vmax); K = 10.0
    log = dict(err=[], vmax=[], viol=0)

    def target(t):                                                    # circle in the y-z plane, 6 cm radius
        a = 2 * np.pi * t / 6.0
        return p_c + 0.06 * np.array([0.0, np.sin(a), 1 - np.cos(a)]), R_t

    def ik_tick(t):
        q = data.qpos[prob.qadr].copy(); p, R = prob.fk(q); p_t, R_tt = target(t)
        e = pose_error6(p, R, p_t, R_tt)
        if method == "dls":   dq = dls_step(prob, q, e, ik_dt, K, vmax=vmax)
        elif method == "qp":  dq = qp_step(prob, q, e, ik_dt, K=K, vmax=vmax)
        else:                 dq = mstep(prob, q, p_t, R_tt, ik_dt, gain=min(1.0, K * ik_dt))
        ctrl.set_velocity_target(dq)
        log["err"].append(np.linalg.norm(e[:3])); log["vmax"].append(np.abs(dq).max())
        log["viol"] += (not prob.within_limits(q, 1e-3))

    steps_per_ik = int(round(ik_dt / model.opt.timestep)); n_ik = int(T / ik_dt)
    if view:
        import time
        from mujoco import viewer as mjv
        with mjv.launch_passive(model, data) as v:
            k = 0
            while v.is_running():
                if k % steps_per_ik == 0: ik_tick(data.time)
                mujoco.mj_step(model, data); k += 1; v.sync(); time.sleep(model.opt.timestep)
    else:
        for k in range(n_ik * steps_per_ik):
            if k % steps_per_ik == 0: ik_tick(data.time)
            mujoco.mj_step(model, data)
    mujoco.set_mjcb_control(None)
    e = np.array(log["err"])
    return dict(rms_mm=1e3 * np.sqrt(np.mean(e[len(e)//4:] ** 2)), max_mm=1e3 * e.max(), peak_dq=max(log["vmax"]), viol=log["viol"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--method", default="all", choices=["dls", "qp", "mink", "all"])
    ap.add_argument("--view", action="store_true"); ap.add_argument("--vmax", type=float, default=1.0); a = ap.parse_args()
    for m in (["dls", "qp", "mink"] if a.method == "all" else [a.method]):
        r = run(m, a.view, vmax=a.vmax); print(f"{m:5s} steady RMS {r['rms_mm']:.2f} mm | max {r['max_mm']:.2f} mm | peak |dq| cmd {r['peak_dq']:.2f} rad/s | limit viol ticks {r['viol']}")
