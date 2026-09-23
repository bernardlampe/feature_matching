"""RANSAC homography estimation (H&Z Algorithm 4.6, with the fixes from
this repo: cardinality-first acceptance, sampling without replacement is
handled by rng.sample, >=4 inlier guard, re-estimation on consensus)."""

import math
import random

import numpy as np


def homography(pts1, pts2):
    """Normalized DLT (H&Z): estimate 3x3 H mapping pts1 -> pts2."""
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)
    x, xp = pts1[:, :2], pts2[:, :2]
    c1, c2 = x.mean(axis=0), xp.mean(axis=0)
    d1 = np.linalg.norm(x - c1, axis=1).mean()
    d2 = np.linalg.norm(xp - c2, axis=1).mean()
    s1, s2 = math.sqrt(2.0) / d1, math.sqrt(2.0) / d2
    T1 = np.array([[s1, 0, -s1 * c1[0]], [0, s1, -s1 * c1[1]], [0, 0, 1]])
    T2 = np.array([[s2, 0, -s2 * c2[0]], [0, s2, -s2 * c2[1]], [0, 0, 1]])
    xn = (x - c1) * s1
    xpn = (xp - c2) * s2

    # H&Z/C++ layout: source coords are the vectors, dest coords the
    # scalars (matches Homography.h: pts2.z * pts1.x, pts2.y * pts1.x, ...)
    rows = []
    for (xx, xy), (xp_, yp_) in zip(xn, xpn):
        rows.append([0, 0, 0, -xx, -xy, -1, yp_ * xx, yp_ * xy, yp_])
        rows.append([xx, xy, 1, 0, 0, 0, -xp_ * xx, -xp_ * xy, -xp_])
    A = np.asarray(rows)
    _, _, Vt = np.linalg.svd(A)
    H = np.linalg.inv(T2) @ Vt[-1].reshape(3, 3) @ T1
    return H / H[2, 2]


def project(h, x, y):
    d = h[2, 0] * x + h[2, 1] * y + h[2, 2]
    if abs(d) < 1e-12:
        return None
    px = (h[0, 0] * x + h[0, 1] * y + h[0, 2]) / d
    py = (h[1, 0] * x + h[1, 1] * y + h[1, 2]) / d
    if not (np.isfinite(px) and np.isfinite(py)):
        return None
    return px, py


def ransac_homography(p1, p2, dist_thresh=5.0, iters=2000, seed=0x442):
    """Standard RANSAC: maximize inlier count, then re-estimate on the
    consensus set. Returns (H, inlier_idx).

    Vectorized: the point arrays are lifted to numpy once; per-iteration
    inlier sets are computed with matrix math instead of a per-point
    Python loop. Sampling, acceptance (cardinality-first), degenerate
    handling, and the final re-estimation match the reference exactly.
    """
    rng = random.Random(seed)
    n = len(p1)
    P1 = np.asarray(p1, dtype=np.float64)
    P2 = np.asarray(p2, dtype=np.float64)
    x1, y1 = P1[:, 0], P1[:, 1]
    x2, y2 = P2[:, 0], P2[:, 1]
    best_inl = []
    best_H = None
    for _ in range(iters):
        sample = rng.sample(range(n), 4)
        try:
            H = homography([p1[i] for i in sample],
                           [p2[i] for i in sample])
        except (np.linalg.LinAlgError, ValueError):
            continue
        try:
            hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue  # degenerate sample (e.g. collinear/near-duplicate)
        # forward: p1 -> H -> compare to p2
        d1 = H[2, 0] * x1 + H[2, 1] * y1 + H[2, 2]
        ok1 = np.abs(d1) > 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            qx = (H[0, 0] * x1 + H[0, 1] * y1 + H[0, 2]) / d1
            qy = (H[1, 0] * x1 + H[1, 1] * y1 + H[1, 2]) / d1
        e1 = np.where(ok1, (qx - x2) ** 2 + (qy - y2) ** 2, np.inf)
        # backward: p2 -> hinv -> compare to p1
        d2 = hinv[2, 0] * x2 + hinv[2, 1] * y2 + hinv[2, 2]
        ok2 = np.abs(d2) > 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            px = (hinv[0, 0] * x2 + hinv[0, 1] * y2 + hinv[0, 2]) / d2
            py = (hinv[1, 0] * x2 + hinv[1, 1] * y2 + hinv[1, 2]) / d2
        e2 = np.where(ok2, (px - x1) ** 2 + (py - y1) ** 2, np.inf)
        inl_mask = (e1 + e2) < dist_thresh ** 2
        cnt = int(inl_mask.sum())
        # cardinality first (fewer outliers wins), no error tie-break needed
        if cnt > len(best_inl):
            best_inl = [i for i in range(n) if inl_mask[i]]
            best_H = H
    if len(best_inl) < 4:
        raise ValueError("RANSAC failed to find a consensus set of >= 4 points")
    return homography(np.array([p1[i] for i in best_inl]),
                      np.array([p2[i] for i in best_inl])), best_inl
