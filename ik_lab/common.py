"""Shared pieces: model loading, Jacobian, 6D pose error, result container."""
import os
from dataclasses import dataclass, field
import numpy as np
import mujoco

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEAL_XML = os.path.join(REPO, "robot_descriptions", "single_arm_heal_effort_actuation_rs_mj.xml")
EE_SITE = "right_center"          # site at the end-effector flange (same one used for gripper attach)


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iters: int
    pos_err: float
    rot_err: float
    traj: list = field(default_factory=list)   # q history (for animation / convergence plots)
    method: str = ""


class IKProblem:
    """Wraps a model+data and exposes FK, Jacobian, limits. Solvers never touch `data` of the
    live simulation: they work on a private MjData so IK can be run 'offline' and then played back."""

    def __init__(self, xml=HEAL_XML, ee_site=EE_SITE, n_arm=6, limit_scale=1.0, model=None):
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site(ee_site).id
        self.n = n_arm
        self.jids = np.arange(n_arm)                       # first n joints are the arm
        self.dofs = self.model.jnt_dofadr[self.jids]
        self.qadr = self.model.jnt_qposadr[self.jids]
        self.set_limit_scale(limit_scale)

    def set_limit_scale(self, s):
        """Shrink joint ranges about their centre (used in the benchmark to stress limits)."""
        r = self.model.jnt_range[self.jids].copy()
        if not hasattr(self, "_r0"): self._r0 = r.copy()
        c, h = self._r0.mean(1), (self._r0[:, 1] - self._r0[:, 0]) / 2 * s
        self.model.jnt_range[self.jids] = np.c_[c - h, c + h]
        self.qmin, self.qmax = self.model.jnt_range[self.jids].T.copy()

    # ---- kinematics via MuJoCo (no hand-derived math needed for HEAL) ----
    def fk(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def jac(self, q):
        """6xn geometric Jacobian [Jv; Jw] expressed in the world frame."""
        self.fk(q); mujoco.mj_comPos(self.model, self.data)   # refresh for jacobian
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.site)
        return np.vstack([jp[:, self.dofs], jr[:, self.dofs]])

    def clip(self, q): return np.clip(q, self.qmin, self.qmax)
    def within_limits(self, q, tol=1e-6): return bool(np.all(q >= self.qmin - tol) and np.all(q <= self.qmax + tol))


def pose_error6(p, R, p_t, R_t):
    """[position error ; orientation error] in the world frame (rotvec of R_t R^T)."""
    from scipy.spatial.transform import Rotation as Rot
    return np.r_[p_t - p, Rot.from_matrix(R_t @ R.T).as_rotvec()]


def make_target(prob, q_goal):
    """Reachable-by-construction target: FK of a known joint vector."""
    return prob.fk(q_goal)


TOL_P, TOL_R = 1e-3, 1e-2     # success: 1 mm, ~0.6 deg
