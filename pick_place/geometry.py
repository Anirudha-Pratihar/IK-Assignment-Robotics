"""Small SE(3) / trajectory helpers shared by the planner, workspace analysis and episode code."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


def wrap(a):
    """wrap angle to [-pi, pi)"""
    return (a + np.pi) % (2 * np.pi) - np.pi


def topdown_R(phi):
    """Tool frame with z pointing DOWN and the tool x-axis (jaw closing axis) at angle phi in the world xy-plane."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def tool_phi(R):
    """angle of the tool x-axis in the world xy-plane"""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def make_T(p, R):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


def min_jerk(tau):
    """s(tau), ds/dtau for tau in [0,1]"""
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5, 30 * tau**2 - 60 * tau**3 + 30 * tau**4


def segment_time(p0, R0, p1, R1, v_cart, w_cart, t_min):
    dist = np.linalg.norm(p1 - p0)
    ang = np.linalg.norm(Rot.from_matrix(R1 @ R0.T).as_rotvec())
    # min-jerk peak speed = 1.875 x average
    return max(t_min, 1.875 * dist / v_cart, 1.875 * ang / w_cart)


def interp_pose(p0, R0, p1, R1, s):
    p = p0 + s * (p1 - p0)
    rv = Rot.from_matrix(R1 @ R0.T).as_rotvec()
    R = Rot.from_rotvec(s * rv).as_matrix() @ R0
    return p, R


def box_overlap_free(x, y, rects, margin=0.0):
    """True if (x,y) is outside all rectangles (xmin,xmax,ymin,ymax) expanded by margin"""
    for (x0, x1, y0, y1) in rects:
        if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin: return False
    return True


def largest_true_rectangle(mask):
    """Largest axis-aligned all-True rectangle in a 2-D boolean array (rows=i, cols=j).
    Returns (area_cells, i0, i1, j0, j1) inclusive.  O(rows*cols) histogram method."""
    rows, cols = mask.shape
    h = np.zeros(cols, dtype=int); best = (0, 0, -1, 0, -1)
    for i in range(rows):
        h = np.where(mask[i], h + 1, 0)
        stack = []
        for j in range(cols + 1):
            cur = h[j] if j < cols else 0
            start = j
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop(); area = hh * (j - s)
                if area > best[0]: best = (area, i - hh + 1, i, s, j - 1)
                start = s
            stack.append((start, cur))
    return best
