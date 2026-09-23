"""Descriptors: normalized patch (simple) and gradient histograms
(simplified SIFT descriptor, Lowe 2004 section 5)."""

import math

import numpy as np
from scipy import ndimage

from .common import gaussian_blur


def patch_descriptor(gray, y, x, half=8, blur=1.0):
    """Normalized 8x8 patch sampled at stride 2 from a blurred
    neighborhood, centered on the keypoint: rows/cols cover
    [y - half, y + half) with the keypoint at the center of the grid
    (mean-subtracted, L2 norm)."""
    H, W = gray.shape
    img = gaussian_blur(gray, blur)
    # stride-2 grid of half*half samples, centered on (y, x): offsets
    # 2*i - half for i in [0, half), i.e. -8..+6 step 2 for half=8
    y0, x0 = int(round(y)), int(round(x))
    patch = np.zeros((half, half))
    for i in range(half):
        for j in range(half):
            yy, xx = y0 + 2 * i - half, x0 + 2 * j - half
            if 0 <= yy < H and 0 <= xx < W:
                patch[i, j] = img[yy, xx]
    p = patch.ravel()
    p = p - p.mean()
    n = np.linalg.norm(p)
    return p / n if n > 1e-8 else p


def sift_like_descriptor(gray, kps, half=8):
    """Gradient-histogram descriptor per keypoint: 4x4 cells x 8
    orientation bins, pooled from blurred gradients, L2-normalized with
    the SIFT 0.2 clamp. Returns (n_kp, 128)."""
    dx = ndimage.sobel(gaussian_blur(gray, 1.0), 1) / 8.0
    dy = ndimage.sobel(gaussian_blur(gray, 1.0), 0) / 8.0
    mag = np.hypot(dx, dy)
    ang = np.arctan2(dy, dx)  # [-pi, pi]
    H, W = gray.shape
    descs = []
    for (y, x, _) in kps:
        # 16px-radius cell window centered on the keypoint: offsets
        # 4*i + a - 2*half, so the (y, x) integer-rounded center is the
        # true geometric center of the 4x4 cell grid
        y0, x0 = int(round(y)) - 2 * half, int(round(x)) - 2 * half
        d = np.zeros((4, 4, 8))
        for i in range(4):
            for j in range(4):
                for a in range(2):
                    for b in range(2):
                        yy, xx = y0 + 4 * i + a + half, x0 + 4 * j + b + half
                        if not (0 <= yy < H and 0 <= xx < W):
                            continue
                        th = ang[yy, xx] + math.pi
                        bin_ = int((th / (2 * math.pi)) * 8) % 8
                        d[i, j, bin_] += mag[yy, xx]
        d = d.ravel()
        n = np.linalg.norm(d)
        if n > 1e-8:
            d = np.minimum(d / n, 0.2)  # SIFT clamps large bin values
            d = d / max(np.linalg.norm(d), 1e-8)
        descs.append(d)
    return np.array(descs) if descs else np.zeros((0, 128))
