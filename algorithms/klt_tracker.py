"""Pyramidal Lucas-Kanade (KLT) point tracker (Lucas & Kanade 1981,
Bouguet 2000 pyramid formulation).

Tracks feature locations from image A to image B by iterative
Gauss-Newton refinement of translation using a windowed spatial-
gradient matrix, coarse-to-fine over a Gaussian pyramid.
"""

import numpy as np
from scipy import ndimage

from .common import gaussian_blur


def _gradients(img):
    gx = ndimage.sobel(img, 1) / 8.0
    gy = ndimage.sobel(img, 0) / 8.0
    return gx, gy


def _bilinear(img, ys, xs):
    """Vectorized bilinear sampling: ys, xs are same-shape float coords;
    returns sampled array of that shape. Out-of-bounds -> 0."""
    H, W = img.shape
    inb = (ys >= 0) & (ys < H - 1) & (xs >= 0) & (xs < W - 1)
    ys_c = np.clip(ys, 0, H - 2)
    xs_c = np.clip(xs, 0, W - 2)
    y0 = ys_c.astype(int)
    x0 = xs_c.astype(int)
    fy = ys_c - y0
    fx = xs_c - x0
    out = (img[y0, x0] * (1 - fx) * (1 - fy) +
           img[y0, x0 + 1] * fx * (1 - fy) +
           img[y0 + 1, x0] * (1 - fx) * fy +
           img[y0 + 1, x0 + 1] * fx * fy)
    return np.where(inb, out, 0.0)


def _bilinear2(img, ys, xs):
    """Bilinear sample of img at float coords ys/xs (same shape grids).
    Returns None if ANY sample touches out-of-bounds: the LK window must
    be fully supported."""
    H, W = img.shape
    out_of_bounds = (ys < 0) | (ys > H - 1) | (xs < 0) | (xs > W - 1)
    if out_of_bounds.any():
        return None
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    # keep floor()+frac exact at the upper edge
    x0 = np.minimum(x0, W - 2)
    y0 = np.minimum(y0, H - 2)
    fy = ys - y0
    fx = xs - x0
    return (img[y0, x0] * (1 - fx) * (1 - fy) +
            img[y0, x0 + 1] * fx * (1 - fy) +
            img[y0 + 1, x0] * (1 - fx) * fy +
            img[y0 + 1, x0 + 1] * fx * fy)


def _lk_step(img1, img2, gx1, gy1, x, y, win, dx_g=0.0, dy_g=0.0,
             max_iter=30, eps=1e-4):
    """One pyramid-level LK for the window centered at (x, y) in img1."""
    half = win // 2
    H, W = img1.shape
    # allow windows whose edge exactly touches the border (the NEAREST
    # legitimate window): the old half+1 guard silently dropped valid
    # border windows (demo lost 2/60 trackable points to it)
    if not (half < x < W - half - 1 and half < y < H - half - 1):
        return None

    # grid for the window (level coords, subpixel-capable)
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    # template + gradients sampled at the TRUE subpixel center via
    # bilinear interpolation: snapping the window to int(round(center))
    # biased every fractional-center track by up to ~0.5 px (systematic
    # 0.3-0.4 px residual observed under pure translation). Bilinear
    # template = exact to first order for pure translations at any
    # fractional center.
    x0, y0 = int(x), int(y)
    # (win+2)^2 neighborhood around the integer corner; sample at
    # fractional offset (fx, fy) within it
    fx, fy = x - x0, y - y0
    I1w = _bilinear2(img1, y0 + yy + fy, x0 + xx + fx)
    if I1w is None:
        return None
    Jx = _bilinear2(gx1, y0 + yy + fy, x0 + xx + fx)
    Jy = _bilinear2(gy1, y0 + yy + fy, x0 + xx + fx)
    if Jx is None or Jy is None:
        return None
    G = np.array([[np.sum(Jx * Jx), np.sum(Jx * Jy)],
                  [np.sum(Jx * Jy), np.sum(Jy * Jy)]])
    if np.linalg.det(G) < 1e-8:
        return None
    Ginv = np.linalg.inv(G)
    dx, dy = dx_g, dy_g

    for _ in range(max_iter):
        ys = (y + yy + dy).ravel()
        xs = (x + xx + dx).ravel()
        Jw = _bilinear(img2, ys, xs)
        err = I1w.ravel() - Jw
        b = np.array([np.sum(Jx.ravel() * err), np.sum(Jy.ravel() * err)])
        eta = Ginv @ b
        dx += eta[0]
        dy += eta[1]
        if abs(eta[0]) + abs(eta[1]) < eps:
            break
    return dx, dy


def klt_track(img1, img2, pts, win_size=15, num_pyr=3):
    """Track pts [(x, y, ...extra)] from img1 to img2. Returns
    [(x2, y2) | None] per input point (None = tracking failed)."""
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)

    # pyramid: index 0 = COARSEST, index num_pyr-1 = finest
    pyr1 = [img1]
    pyr2 = [img2]
    for _ in range(num_pyr - 1):
        pyr1.insert(0, gaussian_blur(pyr1[0], 0.8)[::2, ::2])
        pyr2.insert(0, gaussian_blur(pyr2[0], 0.8)[::2, ::2])
    # index L in the loops below uses pyr[L] = image downscaled 2^L,
    # so invert: coarsest L -> pyr index (num_pyr-1-L)

    grads1 = [_gradients(p) for p in pyr1]
    grads2 = [_gradients(p) for p in pyr2]

    out = []
    for p in pts:
        x, y = float(p[0]), float(p[1])
        dx, dy = 0.0, 0.0
        ok = True
        for L in range(num_pyr - 1, 0, -1):
            idx = num_pyr - 1 - L   # pyramid array index for level L
            scale = 1.0 / 2 ** L
            r = _lk_step(pyr1[idx], pyr2[idx], grads1[idx][0], grads1[idx][1],
                         x * scale, y * scale, win_size,
                         dx_g=dx * scale, dy_g=dy * scale)
            if r is None:
                ok = False
                break
            # each level re-measures the SAME physical displacement in
            # finer units: replace, don't accumulate
            dx = r[0] * 2 ** L
            dy = r[1] * 2 ** L
        if ok:
            r0 = _lk_step(img1, img2, grads1[-1][0], grads1[-1][1],
                          x, y, win_size, dx_g=dx, dy_g=dy)
            if r0 is None:
                ok = False
            else:
                dx = r0[0]
                dy = r0[1]
        out.append((x + dx, y + dy) if ok else None)
    return out
