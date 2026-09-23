"""Essential matrix and relative pose (Module 11).

E = K2^T F K1 Decomposition into R|t per H&Z section 9.6, the cheirality
check which selects which of the four physically possible decompositions
is the correct one, and ideal normalized-camera triangulation.
"""

import numpy as np

from .fundamental import fundamental


def essential_matrix(F, K1, K2):
    """E = K2^T F K1, forced back to its singular-value structure
    (two equal singular values, third zero)."""
    E = K2.T @ F @ K1
    U, S, Vt = np.linalg.svd(E)
    S = np.array([1.0, 1.0, 0.0])  # project to the essential manifold
    E = U @ np.diag(S) @ Vt
    if np.linalg.det(E) < 0:
        E = -E
    return E


def decompose_essential(E):
    """The four (R, t) candidates: R1/R2 x {+t, -t}."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = U @ np.diag([1.0, 1.0, -1.0])
    if np.linalg.det(Vt) < 0:
        Vt = Vt @ np.diag([1.0, 1.0, -1.0])
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def cheirality_check(candidates, p1, p2, K1, K2):
    """Pick the (R, t) under which the most correspondences triangulate
    in front of both cameras (positive depths). Raises on degenerate
    input where NO candidate yields a single positive depth."""
    K1inv = np.linalg.inv(K1)
    K2inv = np.linalg.inv(K2)
    best, best_n = None, 0
    for (R, t) in candidates:
        n = 0
        for a, b in zip(p1, p2):
            x1 = K1inv @ np.r_[a, 1.0]
            x2 = K2inv @ np.r_[b, 1.0]
            if _depth_positive(R, t, x1, x2):
                n += 1
        if n > best_n:
            best, best_n = (R, t), n
    if best is None or best_n == 0:
        raise ValueError("cheirality check found no candidate with "
                         "positive-depth points (degenerate data?)")
    return best, best_n


def _depth_positive(R, t, x1, x2):
    """Triangulate one correspondence from normalized rays; return True
    if both camera depths are positive."""
    # Linear triangulation (H&Z 12.2): P1 = [I|0], P2 = [R|t]
    P1 = np.c_[np.eye(3), np.zeros(3)]
    P2 = np.c_[R, t.reshape(3, 1)]
    a1 = x1[0] * P1[2] - P1[0]
    a2 = x1[1] * P1[2] - P1[1]
    b1 = x2[0] * P2[2] - P2[0]
    b2 = x2[1] * P2[2] - P2[1]
    A = np.vstack([a1, a2, b1, b2])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return False
    X = X[:3] / X[3]
    d1 = X @ x1           # depth in cam1 along its ray direction
    X2 = R @ X + t
    d2 = X2 @ x2          # depth in cam2
    return d1 > 0 and d2 > 0


def estimate_relative_pose(p1, p2, K1, K2, dist_thresh=2.0, iters=2000,
                           seed=0x442):
    """Full two-view pipeline: RANSAC F -> E -> cheirality-selected pose.
    Returns (R, t, inlier_idx)."""
    from . import fundamental as fnd
    F, inl = fnd.ransac_fundamental(p1, p2, dist_thresh=dist_thresh,
                                    iters=iters, seed=seed)
    E = essential_matrix(F, K1, K2)
    (R, t), n = cheirality_check(decompose_essential(E), p1, p2, K1, K2)
    return R, t, inl
