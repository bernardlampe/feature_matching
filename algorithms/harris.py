"""Harris corner detection -- numpy port of assignments/harris/harris.m.

Faithful port including the seven fixes applied to harris.m:
  - scalar rmin/rmax via flattening (min(R(:)) semantics)
  - negative responses zeroed before non-maximum suppression
  - elementwise inverse of the diagonal eigenvalue matrix

Usage: python3 algorithms/harris.py <image> <sigma> <winSize> <threshold> [outdir]
"""
import math
import os
import sys

import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def to_gray(img):
    """RGB/gray uint8 array -> float64 2-D array (MATLAB rgb2gray weights)."""
    if img.ndim == 3:
        return (0.2989 * img[..., 0] + 0.5870 * img[..., 1] +
                0.1140 * img[..., 2])
    return img.astype(np.float64)


def gaussian_kernel(size, sigma):
    """fspecial('gaussian', size, sigma) equivalent."""
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    k = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return k / k.sum()


def harris(fname, sigma, winsize, threshold, outdir="."):
    im = np.asarray(Image.open(fname).convert("RGB"))
    im_gray = to_gray(im)
    height, width = im_gray.shape

    # gaussian smoothing; kernel wide enough for +-2.5 sigma
    ksize = 1 + 2 * int(math.ceil(2.5 * sigma))
    gaus = gaussian_kernel(ksize, sigma)
    # centered derivative filters, replicate boundary (imfilter 'replicate')
    xfilt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)

    def conv_replicate(a, k):
        ph, pw = k.shape[0] // 2, k.shape[1] // 2
        padded = np.pad(a, ((ph, ph), (pw, pw)), mode="edge")
        return convolve2d(padded, k, mode="valid")

    im_gaus = conv_replicate(im_gray, gaus)
    im_dx = conv_replicate(im_gaus, xfilt)
    im_dy = conv_replicate(im_gaus, xfilt.T)


    # squared and cross derivatives
    im_dx2 = im_dx * im_dx
    im_dy2 = im_dy * im_dy
    im_dxdy = im_dx * im_dy

    s = winsize / 5.0
    win = gaussian_kernel(winsize, s)
    sum_dx2 = conv_replicate(im_dx2, win)
    sum_dy2 = conv_replicate(im_dy2, win)
    sum_dxdy = conv_replicate(im_dxdy, win)

    # corner response R = det(M) - k*trace(M)^2, k = 0.04
    k = 0.04
    a, b, c = sum_dx2, sum_dy2, sum_dxdy
    R = a * b - c * c - k * (a + b) * (a + b)

    # threshold with SCALAR min/max (fixed min(R(:)) semantics)
    rmin = R.min()
    rmax = R.max()
    thresh = (rmax - rmin) * threshold + rmin

    # zero out negatives + below-threshold (fix #2)
    im_th = np.where(R > thresh, R, 0.0)
    im_th[im_th < 0.0] = 0.0

    # 3x3 non-maximum suppression == imregionalmax; out-of-bounds counts
    # as -inf so real peaks at the border survive
    m = im_th
    padded = np.pad(m, 1, mode="constant", constant_values=-np.inf)
    peaks = np.ones_like(m, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            peaks &= m > padded[1 + dy:1 + dy + m.shape[0],
                                1 + dx:1 + dx + m.shape[1]]
    ys, xs = np.nonzero(peaks & (im_th > 0.0))
    corners = [(int(y), int(x)) for y, x in zip(ys, xs)]

    _draw_outputs(fname, im_th, corners, outdir)
    return corners


def _draw_outputs(fname, im_th, corners, outdir):
    base = os.path.splitext(os.path.basename(fname))[0]
    os.makedirs(outdir, exist_ok=True)

    # output 1: image with green crosses at corners
    im = np.asarray(Image.open(fname).convert("RGB")).copy()
    h, w = im.shape[:2]
    for (y, x) in corners:
        for d in range(-3, 4):
            if 0 <= y < h and 0 <= x + d < w:
                im[y, x + d] = (0, 255, 0)
            if 0 <= y + d < h and 0 <= x < w:
                im[y + d, x] = (0, 255, 0)
    out1 = os.path.join(outdir, base + "_corners.png")
    Image.fromarray(im).save(out1)

    # output 2: response map rendered as grayscale PNG
    vmax = im_th.max()
    if vmax > 0:
        resp = (255.0 * im_th / vmax + 0.5).astype(np.uint8)
    else:
        resp = np.zeros_like(im_th, dtype=np.uint8)
    out2 = os.path.join(outdir, base + "_response.png")
    Image.fromarray(resp, "L").save(out2)

    print("corners: %d -> %s, %s" % (len(corners), out1, out2))


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        sys.exit("usage: harris.py <image> <sigma> <winSize> <threshold> "
                 "[outdir]")
    harris(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]),
           float(sys.argv[4]), sys.argv[5] if len(sys.argv) == 6 else ".")
