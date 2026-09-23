"""Synthetic-transform exercise suite.

Generates transformed copies of an image together with EXACT ground-truth
correspondences, so descriptor/detector invariance claims can be measured
instead of asserted. Used by the Module assessment exercises (the
"invariance matrix") and by tests/test_invariance.py.

Coordinates: image functions use (y, x) rows/cols internally; the returned
correspondence lists use (x, y) order to match data/correspondences/*.txt
files and the ransac and fundamental modules in this package.
"""

import math

import numpy as np
from scipy import ndimage


def warp_image(im, theta_deg=0.0, scale=1.0, brightness=0.0, contrast=1.0):
    """Photometric + geometric transform of a grayscale image.

    Forward map (source (y, x) -> canvas (y, x)):
        p_out = A @ (p_in - c) + c + off,   A = scale * R(theta),  c = center

    The canvas is grown just enough to contain the warped image.
    Photometrics are applied BEFORE the warp: out = contrast*I + brightness,
    clipped to uint8 range.

    Returns (out, fwd, inv):
      out    -- warped image, (OH, OW) float64
      fwd    -- callable, source (y, x) -> canvas (y, x)
      inv    -- callable, canvas (y, x) -> source (y, x)
    """
    im = np.asarray(im, dtype=np.float64)
    # photometric stage, applied BEFORE the warp as documented
    if brightness != 0.0 or contrast != 1.0:
        im = np.clip(im * contrast + brightness, 0.0, 255.0)
    H, W = im.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    # acting on (row, col): consistent as long as fwd/inv are inverses
    A = np.array([[scale * c, -scale * s], [scale * s, scale * c]])
    center = np.array([cy, cx])

    corners = np.array([[0, 0], [0, W - 1], [H - 1, 0], [H - 1, W - 1]],
                       dtype=np.float64)
    mapped = np.array([A @ (p - center) + center for p in corners])
    lo = np.floor(mapped.min(axis=0)).astype(int)
    hi = np.ceil(mapped.max(axis=0)).astype(int)
    off = -lo
    OH, OW = int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1

    Ainv = np.linalg.inv(A)
    M = Ainv
    offset = center - Ainv @ (center + off)
    out = ndimage.affine_transform(im, M, offset=offset, output_shape=(OH, OW),
                                   order=1, mode="constant", cval=0.0)

    fwd = lambda p: A @ (np.asarray(p, float) - center) + center + off
    inv = lambda q: Ainv @ (np.asarray(q, float) - off - center) + center
    return out, fwd, inv


def gt_correspondences(im, inv, out_shape, n=20, margin=30, seed=1):
    """Sample n ground-truth correspondences for a warped image.

    Picks dest points uniformly in the canvas (inset by `margin`), maps
    them back through `inv`, and keeps those whose source position is
    >= margin inside the source image (so patches are fully supported
    on both sides).

    Returns (src [(x, y)...], dst [(x, y)...]) — pixel-coordinate order,
    matching ransac.py / fundamental.py expectations.
    """
    rng = np.random.RandomState(seed)
    H, W = im.shape
    OH, OW = out_shape
    src, dst = [], []
    tries = 0
    while len(src) < n and tries < 100 * n:
        tries += 1
        q = np.array([rng.uniform(margin, OH - 1 - margin),
                      rng.uniform(margin, OW - 1 - margin)])
        p = inv(q)
        if not (margin <= p[0] < H - margin and margin <= p[1] < W - margin):
            continue
        src.append((p[1], p[0]))
        dst.append((q[1], q[0]))
    return src, dst
