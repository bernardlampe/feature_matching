"""Panorama homography + blending: the classic "stitcher" demo of
homography-based image registration (exercise of HW3/RANSAC applied to
real pixel structure)."""

import numpy as np
from PIL import Image

from .common import load_gray
from .harris_detector import harris_corners
from .descriptors import patch_descriptor
from .matching import match_descriptors
from .ransac import ransac_homography


def homog_warp(H, src, width, height):
    """Warp src (H, W, 3 RGB uint8) by homography H into a
    (height, width, 3) canvas. Bilinear sampling, zero outside."""
    Hi = np.linalg.inv(H)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    p = np.stack([xs, ys, np.ones_like(xs)], axis=-1) @ Hi.T
    den = p[..., 2]
    valid = np.abs(den) > 1e-9
    sx = np.where(valid, p[..., 0] / np.where(valid, den, 1), -1)
    sy = np.where(valid, p[..., 1] / np.where(valid, den, 1), -1)
    inb = (sx >= 0) & (sx < src.shape[1] - 1) & \
          (sy >= 0) & (sy < src.shape[0] - 1)
    sx = np.clip(sx, 0, src.shape[1] - 1)
    sy = np.clip(sy, 0, src.shape[0] - 1)
    x0 = sx.astype(int)
    y0 = sy.astype(int)
    x1 = np.minimum(x0 + 1, src.shape[1] - 1)
    y1 = np.minimum(y0 + 1, src.shape[0] - 1)
    fx = sx - x0
    fy = sy - y0
    out = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        chan = src[..., c]
        top = (1 - fx) * chan[y0, x0] + fx * chan[y0, x1]
        bot = (1 - fx) * chan[y1, x0] + fx * chan[y1, x1]
        val = (1 - fy) * top + fy * bot
        out[..., c] = np.where(inb, np.rint(val), 0).astype(np.uint8)
    return out


def blend(img_a, img_b):
    """Average overlap blend with masks."""
    ma = (img_a.max(axis=2) > 0) if img_a.ndim == 3 else (img_a > 0)
    mb = (img_b.max(axis=2) > 0) if img_b.ndim == 3 else (img_b > 0)
    both = ma & mb
    out = img_a.copy()
    out[both] = (0.5 * img_a[both] + 0.5 * img_b[both]).astype(np.uint8)
    out[mb & ~both] = img_b[mb & ~both]
    return out


def stitch(img_a, img_b, H_ab):
    """img_a and img_b RGB uint8; H_ab maps points of A to B coordinates.
    Output: canvas with A at origin, B warped by H_ab, overlap averaged."""
    h, w = img_a.shape[:2]
    h2, w2 = img_b.shape[:2]
    corners = np.array([[0, 0, 1], [w2, 0, 1], [0, h2, 1], [w2, h2, 1]],
                       dtype=np.float64)
    # B content appears in the canvas at H_ab^-1(B-rect): the canvas
    # (A-frame) coords sampled INTO B. Using the forward map H here
    # over-crops — on the old I1/I3 pair 52% of I3 content fell outside
    # the canvas (post-fix audit probe).
    pos = corners @ np.linalg.inv(H_ab).T
    pos = pos[:, :2] / pos[:, 2:3]
    ox = min(0, int(np.floor(pos[:, 0].min())))
    oy = min(0, int(np.floor(pos[:, 1].min())))
    maxw = max(w, int(np.ceil(pos[:, 0].max()))) - ox
    maxh = max(h, int(np.ceil(pos[:, 1].max()))) - oy
    # A placed at (-ox, -oy); B warped by T(-ox,-oy) @ H so coordinates
    # share the canvas frame
    T = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], dtype=np.float64)
    canvas = np.zeros((maxh, maxw, 3), dtype=np.uint8)
    a = np.zeros_like(canvas)
    a[-oy:-oy + h, -ox:-ox + w] = img_a
    # homog_warp(canvas -> src): pass the INVERSE so B is resampled
    # backwards from the shared canvas frame
    warped_b = homog_warp(T @ np.linalg.inv(H_ab), img_b, maxw, maxh)
    return blend(a, warped_b)
