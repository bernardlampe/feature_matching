"""FAST corner detector (Rosten & Drummond 2006).

Tests 16 pixels on a Bresenham circle of radius 3 around a candidate:
corner if N contiguous pixels are all brighter than I+d (FAST-9..12) or
all darker than I-d. Includes the standard opposite-pair quick-rejection
pre-test and non-max suppression on the segment test score (FAST-ER
lite).

Implementation is fully vectorized (numpy over the whole grid); the
detection decisions, scores, ordering, and NMS match the original
per-pixel reference exactly.
"""

import numpy as np

# FAST-16 Bresenham ring, radius 3: all (dx,dy) with dx^2+dy^2 in {9,10},
# clockwise starting at north. Indices 0/4/8/12 are N/E/S/W for the
# opposite-pair pre-test.
CIRCLE16 = [(0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1),
            (2, 2), (1, 3), (0, 3), (-1, 3), (-2, 2), (-3, 1),
            (-3, 0), (-3, -1), (-2, -2), (-1, -3)]


def fast_corners(gray, n=9, threshold=20.0):
    """Returns list of (y, x, score) sorted by score desc."""
    if n < 9 or n > 12:
        raise ValueError("FAST supports n in [9, 12]; got %d" % n)
    img = np.ascontiguousarray(gray, dtype=np.int16)
    H, W = img.shape
    if H < 7 or W < 7:
        return []
    # candidate grid: same extent as the reference loop (3 .. H-4)
    ys, xs = np.mgrid[3:H - 3, 3:W - 3]
    c = img[3:H - 3, 3:W - 3]
    # 16 ring slices, each aligned with the candidate grid
    ring = np.stack([img[3 + dy:H - 3 + dy, 3 + dx:W - 3 + dx]
                     for dy, dx in CIRCLE16])  # (16, Rh, Rw)
    hi = ring > c[None] + threshold
    lo = ring < c[None] - threshold
    # quick-reject: a contiguous 9-arc on a 16-ring must include at
    # least one pixel of EACH opposite pair (i, i+8) per polarity
    hi_ok = np.ones(c.shape, dtype=bool)
    lo_ok = np.ones(c.shape, dtype=bool)
    for i in range(8):
        hi_ok &= hi[i] | hi[i + 8]
        lo_ok &= lo[i] | lo[i + 8]
    cand = hi_ok | lo_ok
    cy, cx = np.nonzero(cand)
    if cy.size == 0:
        return []
    # segment score for candidates only: longest contiguous True run on
    # the cyclic 16-ring (capped at 16), per polarity
    hi_c = np.stack([hi[i][cy, cx] for i in range(16)], axis=1)  # (N,16)
    lo_c = np.stack([lo[i][cy, cx] for i in range(16)], axis=1)
    score = np.maximum(_run_scores(hi_c), _run_scores(lo_c))
    score = np.where(score >= n, score, 0)
    keep = score > 0
    cy, cx, score = cy[keep], cx[keep], score[keep]
    cy = cy + 3  # candidate grid starts at row/col 3: make absolute
    cx = cx + 3
    # stable sort by -score; candidates are already in row-major order,
    # so ties keep the reference scan order
    order = np.argsort(-score, kind="stable")
    cy, cx, score = cy[order], cx[order], score[order]
    # non-max suppression on a 3x3 neighborhood (greedy, score order)
    occupied = np.zeros((H, W), dtype=bool)
    pts = []
    for y, x, s in zip(cy.tolist(), cx.tolist(), score.tolist()):
        if not occupied[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any():
            kept = (y, x, s)
            pts.append(kept)
            occupied[y, x] = True
    return pts


def _run_scores(hits):
    """(N, 16) bool -> longest cyclic True run per row, capped at 16.

    For each of the 16 start offsets, the run length is the count of
    leading Trues in the 16-entry window starting there (a longer true
    run wraps and is caught by the offset where it begins; the cap at 16
    matches the reference, where a full ring scores 16).
    """
    N = hits.shape[0]
    ring2 = np.concatenate([hits, hits], axis=1)  # (N, 32)
    best = np.zeros(N, dtype=np.int32)
    for s in range(16):
        win = ring2[:, s:s + 16]
        lead = np.where(win.all(axis=1), 16,
                        np.argmin(win, axis=1))  # first False index
        np.maximum(best, lead, out=best)
    return best


def _segment_score(hits, n):
    """Scalar reference path (kept for tests/debug on one ring)."""
    ring2 = np.concatenate([hits, hits])
    run = best = 0
    for v in ring2:
        run = run + 1 if v else 0
        if run > best:
            best = run
    if best > 16:
        best = 16  # full ring: all 16 True
    return best if best >= n else 0
