"""Normalized-cross-correlation patch matcher (classic pre-BRIEF
intensity matcher; compares patches in a search region rather than via
precomputed descriptor vectors).
"""

import numpy as np
from scipy import ndimage

from .common import gaussian_blur


def _patch_ncc(a, b):
    """NCC between two equal-size patches."""
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    if d < 1e-9:
        return -1.0
    return float(np.dot(a, b) / d)


def ncc_match(img1, img2, pts1, search_radius=25, win=21):
    """For each point in pts1 search a +-search_radius window in img2 for
    the maximum NCC patch. Returns [(x2, y2, ncc)] per point — the raw
    best score with NO quality threshold (None only when the window
    falls off the image borderline). Filtering by score is the caller's
    job; the Module 5 near-repeat demo shows why a high NCC alone is
    not confidence (use the ratio test / mutual check instead)."""
    im1 = gaussian_blur(img1, 1.0)
    im2 = gaussian_blur(img2, 1.0)
    half = win // 2
    H, W = img2.shape
    out = []
    for p1 in pts1:
        x, y = int(round(p1[0])), int(round(p1[1]))
        if not (half < x < W - half and half < y < H - half):
            out.append(None)
            continue
        t = im1[y - half:y + half + 1, x - half:x + half + 1]
        y0, y1 = max(half, y - half - search_radius), \
            min(H - half, y + half + search_radius + 1)
        x0, x1 = max(half, x - half - search_radius), \
            min(W - half, x + half + search_radius + 1)
        best = (-2.0, None, None)
        for yy in range(y0, y1):
            for xx in range(x0, x1):
                w = im2[yy - half:yy + half + 1, xx - half:xx + half + 1]
                s = _patch_ncc(t, w)
                if s > best[0]:
                    best = (s, xx, yy)
        out.append(best if best[1] is not None else None)
    return out
