#!/usr/bin/env python3
"""Standalone panorama stitcher built on the course algorithms.

Chains the from-scratch implementations in algorithms/: FAST corners +
BRIEF descriptors + Hamming matching + RANSAC homography per adjacent
image pair, composes the homographies into the first image's frame, and
blends every image onto one canvas (count-weighted average of overlaps,
the N-image generalization of stitcher.blend).

Usage:
  python3 stitch_tool.py OUT.png IMG1 IMG2 [IMG3 ...] [--show-matches]
  python3 stitch_tool.py OUT.png --hpatches-seq i_lionnight [--frames 1,2]
                                 [--hpatches-root DIR]

Options:
  --max-kps N       FAST keypoints per image, capped (default 400)
  --min-inliers N   minimum RANSAC inliers per pair (default 8)
  --seed N          RANSAC seed (default 305419896 = 0x442)
  --show-matches    write matches_pair<k>.png canvases next to OUT.png
  --show-correspondences
                    additionally write OUT_corr.png: the input images
                    side by side with every match drawn as two dots
                    joined by a line (green = RANSAC inlier, red =
                    rejected)
  --hpatches-seq N  take frames from an HPatches sequence instead of
                    explicit image files (--frames picks/ orders them)
  --hpatches-root D HPatches dataset root (default data/hpatches-sequences-release)

Any PIL-readable format works for explicit images (png/jpg/ppm/...).

Exit codes: 0 ok; 2 usage/input error; 3 stitching failure (a pair
produced too few matches or RANSAC inliers).

Requires the algorithms package importable (run from the repo root or
with the repo on PYTHONPATH).
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image

from algorithms.brief_descriptor import brief_descriptors, match_brief
from algorithms.fast_detector import fast_corners
from algorithms.ransac import project, ransac_homography
from algorithms.stitcher import homog_warp
from algorithms.visualize import draw_matches, line, point


def to_gray(im_rgb):
    """RGB uint8 -> float64 grayscale (MATLAB rgb2gray weights)."""
    return (0.2989 * im_rgb[..., 0] + 0.5870 * im_rgb[..., 1] +
            0.1140 * im_rgb[..., 2])


def load_rgb(path):
    im = np.asarray(Image.open(path).convert("RGB"))
    if im.ndim != 3 or im.shape[2] != 3:
        raise ValueError("%s: not an RGB image" % path)
    return im


def corners_of(w, h):
    return np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]],
                    dtype=np.float64)


def multiscale_kps(gray, max_kps, threshold=30.0, scales=(1.0, 0.75, 0.5)):
    """FAST + BRIEF over a scale pyramid.

    BRIEF has no scale compensation (that's ORB's contribution), so a
    keypoint only matches when both views see the patch at a similar
    scale. Detect at several downsamplings, describe at the same scale,
    and lift coordinates back to full-frame (x * 1/s, y * 1/s, s).
    Per scale level the strongest max_kps keypoints are taken; the
    caller's budget applies per level, not globally.
    Returns (descriptors, kps) with kps as (y, x, scale) tuples in
    full-frame coordinates.
    """
    from scipy.ndimage import zoom
    descs, kps = [], []
    occupied = set()
    for s in scales:
        if s == 1.0:
            g = gray
        else:
            g = zoom(gray, s, order=1)
        found = fast_corners(g, threshold=threshold)[:max_kps]
        if not found:
            continue
        d, kk, _ = brief_descriptors(g, [(y, x) for y, x, _ in found])
        kept, kept_d = [], []
        for row, (y, x) in zip(d, kk):
            # dedupe: same corner detected at multiple scales would be
            # two near-identical points in full-frame coords, and a
            # RANSAC sample drawn on them is singular
            key = (int(round(y / s)), int(round(x / s)))
            if key in occupied:
                continue
            occupied.add(key)
            kept.append((y / s, x / s, s))
            kept_d.append(row)
        kps.extend(kept)
        descs.append(np.array(kept_d, dtype=np.uint8))
    all_d = np.concatenate(descs, axis=0) if descs else \
        np.zeros((0, 32), dtype=np.uint8)
    return all_d, kps


def pair_homography(g1, g2, outdir, pair_no, max_kps, min_inliers, seed,
                    show_matches, feat1=None, feat2=None):
    """Match image pair (g1, g2) and RANSAC a homography g1 -> g2.

    Keypoints come from a scale pyramid (multiscale_kps) so BRIEF only
    ever compares patches sampled at similar scales. feat1/feat2 let the
    caller pass cached (descs, kps) for repeated frames in a chain;
    otherwise they are computed here.

    Returns (H, stats dict). Raises StitchError when matching or RANSAC
    fails the quality bar.
    """
    d1, k1 = feat1 if feat1 is not None else multiscale_kps(g1, max_kps)
    d2, k2 = feat2 if feat2 is not None else multiscale_kps(g2, max_kps)
    matches = match_brief(d1, d2)
    if len(matches) < 4:
        raise StitchError(
            "pair %d: only %d BRIEF matches (need >= 4) -- images likely "
            "do not overlap or the appearance change is too extreme"
            % (pair_no, len(matches)))
    p1 = [(k1[i][1], k1[i][0]) for i, _, _ in matches]
    p2 = [(k2[j][1], k2[j][0]) for _, j, _ in matches]
    try:
        H, inl = ransac_homography(p1, p2, dist_thresh=5.0, seed=seed)
    except ValueError as e:
        raise StitchError("pair %d: RANSAC failed (%s)" % (pair_no, e))
    if len(inl) < min_inliers:
        raise StitchError(
            "pair %d: only %d RANSAC inliers (need >= %d) -- the "
            "matches look inconsistent, refusing to stitch"
            % (pair_no, len(inl), min_inliers))
    # residual of the accepted consensus, as a sanity number
    errs = []
    for i in inl:
        q = project(H, p1[i][0], p1[i][1])
        if q is not None:
            errs.append(math.hypot(q[0] - p2[i][0], q[1] - p2[i][1]))
    med = float(np.median(errs)) if errs else float("inf")
    if show_matches:
        draw_matches("pair%d" % pair_no,
                     [(y, x, 0.0) for y, x, _ in k1],
                     [(y, x, 0.0) for y, x, _ in k2],
                     matches, inl, g1, g2, outdir)
    inl_set = set(inl)
    return H, {"matches": len(matches), "inliers": len(inl), "median": med,
               # matched points in each image's own frame, plus the
               # RANSAC verdict per match index (True = inlier)
               "pts_a": p1, "pts_b": p2,
               "inlier_mask": [i in inl_set for i in range(len(matches))]}


class StitchError(Exception):
    pass


def refine_homographies(Hs, stats):
    """Jointly refine the frame-0 -> frame-m homographies.

    Each pair's inlier matches constrain its two adjacent frames: the
    left point lifted into frame 0 via Hs[k]^-1 must coincide with the
    right point lifted via Hs[k+1]^-1 (Hs[m] maps frame 0 into frame
    m, as consumed by the warp). Solving over all frames at once
    distributes residual error across the chain instead of letting it
    accumulate toward the last frame (which surfaced as ghosting in
    the blended overlaps). Returns the refined Hs.
    """
    from scipy.optimize import least_squares

    def unpack(x):
        out = [np.eye(3)]
        for m in range(1, len(Hs)):
            p = x[8 * (m - 1):8 * m]
            out.append(np.array([[p[0], p[1], p[2]],
                                 [p[3], p[4], p[5]],
                                 [p[6], p[7], 1.0]]))
        return out

    def pair_residuals(Hr):
        res = []
        for k, st in enumerate(stats):
            Ha, Hb = np.linalg.inv(Hr[k]), np.linalg.inv(Hr[k + 1])
            for (ax, ay), (bx, by), inl in zip(st["pts_a"], st["pts_b"],
                                               st["inlier_mask"]):
                if not inl:
                    continue
                qa = project(Ha, ax, ay)
                qb = project(Hb, bx, by)
                if qa is None or qb is None:
                    continue
                res.extend([qa[0] - qb[0], qa[1] - qb[1]])
        return res

    def residuals(x):
        return np.array(pair_residuals(unpack(x)))

    def chain_median(Hr):
        res = pair_residuals(Hr)
        errs = [math.hypot(res[i], res[i + 1]) for i in range(0, len(res), 2)]
        return float(np.median(errs)) if errs else 0.0

    x0 = []
    for H in Hs[1:]:
        Hn = H / H[2, 2]
        x0.extend(Hn[:2, :].reshape(-1).tolist())
        x0.extend(Hn[2, :2].tolist())
    before = chain_median(Hs)
    sol = least_squares(residuals, np.array(x0), method="trf",
                        loss="soft_l1", f_scale=2.0)
    Hs_r = unpack(sol.x)
    print("[refine] chain median residual %.2f px -> %.2f px"
          % (before, chain_median(Hs_r)))
    return Hs_r


def draw_correspondences(images, stats):
    """Draw matches on a side-by-side canvas of the input images.

    For each pair, the two inputs are placed side by side and every
    match is drawn as two dots joined by a line (green = RANSAC inlier,
    red = rejected). Multiple pairs are stacked vertically. Returns the
    canvas; the saved panorama stays clean.
    """
    GREEN, RED = (0, 255, 0), (255, 0, 0)
    canvases = []
    for st, im_a, im_b in zip(stats, images, images[1:]):
        h1, w1 = im_a.shape[:2]
        h2, w2 = im_b.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = im_a
        canvas[:h2, w1:w1 + w2] = im_b
        for (ax, ay), (bx, by), is_inl in zip(st["pts_a"], st["pts_b"],
                                              st["inlier_mask"]):
            color = GREEN if is_inl else RED
            point(canvas, int(round(ax)), int(round(ay)), color, r=2)
            point(canvas, int(round(w1 + bx)), int(round(by)), color, r=2)
            line(canvas, ax, ay, w1 + bx, by, color, thick=0)
        canvases.append(canvas)
    if len(canvases) == 1:
        return canvases[0]
    w = max(c.shape[1] for c in canvases)
    h = sum(c.shape[0] for c in canvases)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    y = 0
    for c in canvases:
        out[y:y + c.shape[0], :c.shape[1]] = c
        y += c.shape[0]
    return out


def stitch_images(images, max_kps=400, min_inliers=8, seed=0x442,
                  show_matches=False, outdir="."):
    """Stitch a list of RGB uint8 images (left/first is the reference).

    Returns (panorama RGB uint8, per-pair stats list)."""
    grays = [to_gray(im) for im in images]
    # Hs[m] maps image m's pixel coordinates into image 0's frame
    Hs = [np.eye(3)]
    stats = []
    feats = [None] * len(images)  # per-frame (descs, kps), computed once
    for k in range(len(images) - 1):
        if feats[k] is None:
            feats[k] = multiscale_kps(grays[k], max_kps)
        if feats[k + 1] is None:
            feats[k + 1] = multiscale_kps(grays[k + 1], max_kps)
        H, st = pair_homography(grays[k], grays[k + 1], outdir, k + 1,
                                max_kps, min_inliers, seed, show_matches,
                                feat1=feats[k], feat2=feats[k + 1])
        # H maps frame k -> frame k+1; compose: frame 1 -> k+1
        Hs.append(H @ Hs[-1])
        st["pair"] = k + 1
        stats.append(st)
        print("[pair %d] matches=%d inliers=%d median residual %.2f px"
              % (k + 1, st["matches"], st["inliers"], st["median"]))

    # joint refinement: distribute per-pair residual across the whole
    # chain instead of accumulating it into the last frames (ghosting)
    Hs = refine_homographies(Hs, stats)

    # global canvas bounds: every image's corners mapped into frame 0
    ox = oy = 0
    maxx = maxy = -float("inf")
    for im, H in zip(images, Hs):
        h, w = im.shape[:2]
        pos = corners_of(w, h) @ H.T
        pos = pos[:, :2] / pos[:, 2:3]
        ox = min(ox, int(np.floor(pos[:, 0].min())))
        oy = min(oy, int(np.floor(pos[:, 1].min())))
        maxx = max(maxx, pos[:, 0].max())
        maxy = max(maxy, pos[:, 1].max())
    w0, h0 = images[0].shape[1], images[0].shape[0]
    maxw = max(w0, int(np.ceil(maxx))) - ox
    maxh = max(h0, int(np.ceil(maxy))) - oy

    # warp every image into the shared canvas; blend with feathered
    # weights (distance to the image border) so overlaps cross-fade
    # smoothly instead of double-exposing misaligned edges
    total = np.zeros((maxh, maxw, 3), dtype=np.float64)
    weight = np.zeros((maxh, maxw), dtype=np.float64)
    for im, H in zip(images, Hs):
        # warp into the shared canvas, offsetting by (-ox, -oy)
        warped = homog_warp(
            np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]],
                     dtype=np.float64) @ np.linalg.inv(H),
            im, maxw, maxh).astype(np.float64)
        mask = warped.max(axis=2) > 0
        # feather weight: distance to the image's valid-region border,
        # capped so wide empty margins don't dominate
        from scipy.ndimage import distance_transform_edt
        w = np.minimum(distance_transform_edt(mask), 32.0)
        total += warped * w[..., None]
        weight += w
    out = np.zeros((maxh, maxw, 3), dtype=np.uint8)
    seen = weight > 0
    out[seen] = np.clip(total[seen] / weight[seen, None], 0, 255).astype(
        np.uint8)
    return out, stats


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stitch images into a panorama using the course's "
                    "FAST + BRIEF + RANSAC implementations.",
        epilog="example: python3 stitch_tool.py pano.png "
               "--hpatches-seq v_adam --frames 1,2,3")
    ap.add_argument("output", help="output panorama path (png/jpg/...)")
    ap.add_argument("images", nargs="*",
                    help="input images in stitch order (>= 2), OR use "
                         "--hpatches-seq")
    ap.add_argument("--hpatches-seq", metavar="NAME", default=None,
                    help="stitch frames of an HPatches sequence")
    ap.add_argument("--hpatches-root", metavar="DIR",
                    default=os.path.join("data", "hpatches-sequences-release"))
    ap.add_argument("--frames", metavar="a,b,...", default="1,2",
                    help="frame indices for --hpatches-seq (default 1,2)")
    ap.add_argument("--max-kps", type=int, default=400)
    ap.add_argument("--min-inliers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0x442)
    ap.add_argument("--show-matches", action="store_true",
                    help="write matches_pair<k>.png next to the output")
    ap.add_argument("--show-correspondences", action="store_true",
                    help="draw the input images side by side with dots + "
                         "connecting lines for every match (green inlier / "
                         "red outlier); writes <out>_corr.png")
    args = ap.parse_args(argv)

    outdir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(outdir, exist_ok=True)

    if args.hpatches_seq:
        from algorithms import hpatches
        frames = [int(t) for t in args.frames.split(",")]
        if len(frames) < 2:
            ap.error("--frames needs at least 2 indices")
        seq = hpatches.load_sequence(args.hpatches_seq, root=args.hpatches_root)
        # load every requested frame (load_sequence reads idx1/idx2 only,
        # so load the remaining frames directly)
        rgbs = {seq["idx1"]: seq["im1rgb"], seq["idx2"]: seq["im2rgb"]}
        from algorithms.hpatches import load_rgb_ppm
        d = os.path.join(args.hpatches_root, args.hpatches_seq)
        for f in frames:
            if f not in rgbs:
                rgbs[f] = load_rgb_ppm(os.path.join(d, "%d.ppm" % f))
        images = [rgbs[f] for f in frames]
        print("sequence %s, frames %s" % (args.hpatches_seq, frames))
    else:
        if len(args.images) < 2:
            ap.error("need at least 2 input images (or --hpatches-seq)")
        images = [load_rgb(p) for p in args.images]

    pano, stats = stitch_images(images, max_kps=args.max_kps,
                                min_inliers=args.min_inliers,
                                seed=args.seed,
                                show_matches=args.show_matches,
                                outdir=outdir)
    Image.fromarray(pano).save(args.output)
    print("[stitch] %d images -> %s (%d x %d px)"
          % (len(images), args.output, pano.shape[1], pano.shape[0]))
    if args.show_correspondences:
        root, ext = os.path.splitext(args.output)
        corr_path = root + "_corr" + (ext if ext else ".png")
        corr = draw_correspondences(images, stats)
        Image.fromarray(corr).save(corr_path)
        print("[stitch] correspondences -> %s" % corr_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except StitchError as e:
        print("error: %s" % e, file=sys.stderr)
        sys.exit(3)
    except FileNotFoundError as e:
        print("error: %s" % e, file=sys.stderr)
        sys.exit(2)
