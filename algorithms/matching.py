"""Nearest-neighbor descriptor matching with the Lowe ratio test and
optional mutual-best (cross-check) filtering (Lowe 2004 section 4.1)."""

import math

import numpy as np


def match_descriptors(d1, d2, ratio=0.85, mutual=True):
    """For each descriptor in d1 find the 2 NNs in d2; accept if
    d(best)/d(second) < ratio. Optionally require mutual best, i.e. the
    true cross-check: (i, j) survives iff j is i's NN *and* i is j's NN
    (both directions ratio-screened over the full distance matrix —
    filtering only over already-accepted pairs drops valid matches;
    audit probe case: 3-row A vs 1-column B lost its true match).

    Vectors are L2-normalized internally, so callers may pass raw
    descriptors; without normalization the 2 - 2ab identity would
    silently compute cosine similarity instead of L2 for non-unit input.
    Returns list of (i1, i2, distance)."""
    if len(d1) == 0 or len(d2) == 0:
        return []
    d1 = np.atleast_2d(np.asarray(d1, dtype=np.float64))
    d2 = np.atleast_2d(np.asarray(d2, dtype=np.float64))
    n1 = np.linalg.norm(d1, axis=1, keepdims=True)
    n2 = np.linalg.norm(d2, axis=1, keepdims=True)
    d1 = d1 / np.maximum(n1, 1e-12)
    d2 = d2 / np.maximum(n2, 1e-12)
    dd = 2.0 - 2.0 * d1 @ d2.T  # squared euclidean (unit vectors)
    if len(d2) < 2:
        # single candidate: no second-best exists, ratio test degenerates;
        # treat second as infinitely far (accept if it is mutual)
        j = np.argmin(dd, axis=1)
        matches = [(i, int(j[i]), math.sqrt(max(dd[i, j[i]], 0.0)))
                   for i in range(len(d1))]
    else:
        best2 = np.argsort(dd, axis=1)[:, :2]
        matches = []
        for i, (j1, j2) in enumerate(best2):
            d_best, d_second = dd[i, j1], dd[i, j2]
            # zero second distance => exact duplicate candidate: fully
            # ambiguous, reject (the old `d_second <= 0` branch ACCEPTED
            # these, defeating the ratio test's purpose)
            if d_second > 0 and d_best / d_second < ratio:
                matches.append((i, int(j1), math.sqrt(max(d_best, 0))))
    if mutual:
        # true cross-check: reverse NN over the FULL matrix, then
        # intersect with ratio-accepted forward matches
        rev_nn = np.argmin(dd, axis=0)
        matches = [(i, j, d) for i, j, d in matches if rev_nn[j] == i]
    return matches
