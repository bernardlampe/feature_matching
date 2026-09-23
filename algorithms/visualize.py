"""Visualization helpers for match canvases and annotated images."""

import os

import numpy as np
from PIL import Image


def line(rgb, x0, y0, x1, y1, color, thick=1):
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for t in range(steps + 1):
        x = int(round(x0 + (x1 - x0) * t / max(steps, 1)))
        y = int(round(y0 + (y1 - y0) * t / max(steps, 1)))
        H, W = rgb.shape[:2]
        for dy in range(-thick, thick + 1):
            for dx in range(-thick, thick + 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    rgb[yy, xx] = color


def point(rgb, x, y, color, r=6):
    H, W = rgb.shape[:2]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    rgb[yy, xx] = color


def draw_matches(name, kp1, kp2, matches, inl_idx, im1, im2, outdir):
    """Side-by-side canvas, one line per match (green inlier, red not)."""
    h1, w1 = im1.shape
    h2, w2 = im2.shape
    h, w = max(h1, h2), w1 + w2
    canvas = np.zeros((h, w))
    canvas[:h1, :w1] = im1
    canvas[:h2, w1:] = im2
    rgb = np.dstack([canvas, canvas, canvas]).astype(np.uint8)
    for k, (i, j, d) in enumerate(matches):
        y1, x1 = kp1[i][0], kp1[i][1]
        y2, x2 = kp2[j][0], kp2[j][1]
        color = (0, 255, 0) if k in inl_idx else (255, 0, 0)
        line(rgb, x1, y1, w1 + x2, y2, color)
    Image.fromarray(rgb).save(os.path.join(outdir, "matches_%s.png" % name))
