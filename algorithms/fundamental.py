"""Normalized 8-point algorithm for the fundamental matrix
(Hartley 1997 / H&Z ch. 11) plus RANSAC on F.

The fundamental matrix F relates two views of a planar or general
scene: x2^T F x1 = 0. Unlike a homography, F has 7 DOF and describes
epipolar geometry (no depth restriction).
"""

import math
import random

import numpy as np


def fundamental(pts1, pts2):
    """Normalized 8-point: least-squares F from >= 8 correspondences.
    Points are (x, y)/image pixels."""
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    def norm(points):
        c = points.mean(axis=0)
        d = np.linalg.norm(points - c, axis=1).mean()
        s = math.sqrt(2.0) / d
        T = np.array([[s, 0, -s * c[0]],
                      [0, s, -s * c[1]],
                      [0, 0, 1]])
        return T

    T1, T2 = norm(pts1), norm(pts2)
    x1n = np.c_[pts1, np.ones(len(pts1))] @ T1.T
    x2n = np.c_[pts2, np.ones(len(pts2))] @ T2.T

    # constraint rows [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    rows = []
    for p, q in zip(x1n, x2n):
        rows.append(np.array([q[0] * p[0], q[0] * p[1], q[0],
                              q[1] * p[0], q[1] * p[1], q[1],
                              p[0], p[1], 1.0]))
    A = np.asarray(rows, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # enforce rank 2 (epipolar constraint)
    U, S, Vt2 = np.linalg.svd(F)
    S[2] = 0.0
    F = U @ np.diag(S) @ Vt2

    # denormalize
    F = T2.T @ F @ T1
    return F / np.linalg.norm(F)


def epipolar_distance(F, p1, p2):
    """Symmetric epipolar point-line distance for one correspondence.
    p = (x, y, 1)."""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    l2 = F @ p1          # epipolar line in image 2
    l1 = F.T @ p2        # epipolar line in image 1
    d2 = abs(p2 @ l2) / math.hypot(l2[0], l2[1]) if np.hypot(l2[0], l2[1]) > 1e-12 else math.inf
    d1 = abs(p1 @ l1) / math.hypot(l1[0], l1[1]) if np.hypot(l1[0], l1[1]) > 1e-12 else math.inf
    return d1 + d2


def ransac_fundamental(p1, p2, dist_thresh=2.0, iters=2000, seed=0x442):
    """RANSAC F estimation with the 7-point sample (8-point with the
    8th as backup). Returns (F, inlier_idx)."""
    rng = random.Random(seed)
    n = len(p1)
    assert n >= 8, "need >= 8 correspondences"
    best_inl, best_F = [], None
    for _ in range(iters):
        sample = rng.sample(range(n), 8)
        try:
            F = fundamental([p1[i] for i in sample],
                            [p2[i] for i in sample])
        except np.linalg.LinAlgError:
            continue
        inl = [i for i in range(n)
               if epipolar_distance(F, np.r_[p1[i], 1.0],
                                    np.r_[p2[i], 1.0]) < dist_thresh]
        if len(inl) > len(best_inl):
            best_inl, best_F = inl, F
    if len(best_inl) < 8:
        raise ValueError("RANSAC-F failed")
    F = fundamental([p1[i] for i in best_inl], [p2[i] for i in best_inl])
    return F, best_inl
