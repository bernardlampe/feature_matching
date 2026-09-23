"""Harris corner detector.

Port of assignments/harris/harris.m including its fixes:
  - scalar rmin/rmax via flattening (min(R(:)) semantics)
  - negative responses zeroed before non-maximum suppression
"""

import math

import numpy as np
from scipy import ndimage

from .common import gaussian_blur, gaussian_kernel1d


def harris_corners(gray, sigma=2.0, winsize=7, k=0.04, rel_thresh=0.01):
    """Corner response R = det(M) - k trace(M)^2; returns sorted (y, x, R)."""
    s = winsize / 5.0
    im_gaus = gaussian_blur(gray, sigma)
    dx = ndimage.sobel(im_gaus, 1) / 8.0
    dy = ndimage.sobel(im_gaus, 0) / 8.0
    win = gaussian_kernel1d(s)[None, :] * gaussian_kernel1d(s)[:, None]
    a = ndimage.convolve(dx * dx, win, mode="nearest")
    b = ndimage.convolve(dy * dy, win, mode="nearest")
    c = ndimage.convolve(dx * dy, win, mode="nearest")
    R = a * b - c * c - k * (a + b) ** 2

    # scalar min/max across the whole array (fixed min(R(:)) semantics),
    # negatives zeroed before NMS (fix #2)
    thresh = (R.max() - R.min()) * rel_thresh + R.min()
    im_th = np.where(R > thresh, R, 0.0)
    im_th[im_th < 0.0] = 0.0

    # 3x3 non-max suppression, -inf outside
    padded = np.pad(im_th, 1, mode="constant", constant_values=-np.inf)
    peaks = np.ones_like(im_th, dtype=bool)
    for dy_ in (-1, 0, 1):
        for dx_ in (-1, 0, 1):
            if dy_ == 0 and dx_ == 0:
                continue
            peaks &= im_th > padded[1 + dy_:1 + dy_ + im_th.shape[0],
                                    1 + dx_:1 + dx_ + im_th.shape[1]]
    ys, xs = np.nonzero(peaks & (im_th > 0.0))
    vals = im_th[ys, xs]
    order = np.argsort(-vals)
    return list(zip(ys[order], xs[order], vals[order]))
