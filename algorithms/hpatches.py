"""HPatches sequence loader (Modules 1–11 data source).

HPatches (hpatches.github.io, Balnties et al. CVPR 2017) ships 116
sequences; each has images ``1.ppm`` .. ``6.ppm`` plus ground-truth
homographies ``H_1_2`` .. ``H_1_6``. Verified convention (photo-
consistency probe on v_wall, v_adam, v_beyus, v_wapping — MAE 9.8-19.9
forward vs 24-88 reversed): **H_1_k maps image-1 pixel coordinates to
image-k pixel coordinates**, matching ransac.py's ``project`` and
stitcher.py's ``H_ab`` (A-frame -> B-frame) conventions directly.

Sequence families (filename prefix):
  i_ -- illumination changes; all H_1_k are the identity, so the GT
        homography trivially aligns the pair but pixel appearance moves.
        This replaces the old pts2.txt "10 deliberately displaced
        outliers" lesson with a real outlier model: only 83-90% of
        detector/descriptor matches survive extreme illumination.
  v_ -- viewpoint changes; H_1_k are genuine projective warps.

Notes
-----
ppm images vary in size within a sequence (e.g. i_lionday 1.ppm is
512x384, 2..6.ppm are 500x375), so every consumer must handle differing
H, W. ``H_1_1`` files exist in some mirrors but not this one; the loader
uses the identity for k == 1.
"""

import os

import numpy as np
from PIL import Image

# Default dataset location relative to the repo root (course_demo.py and
# tests run from there).
DEFAULT_ROOT = os.path.join("data", "hpatches-sequences-release")


def load_gray_ppm(path):
    """ppm -> float64 grayscale (MATLAB rgb2gray weights, like common.py)."""
    im = np.asarray(Image.open(path).convert("RGB"))
    return 0.2989 * im[..., 0] + 0.5870 * im[..., 1] + 0.1140 * im[..., 2]


def load_rgb_ppm(path):
    """ppm -> HxWx3 RGB uint8."""
    return np.asarray(Image.open(path).convert("RGB"))


def load_homography(path):
    """H_1_k file -> 3x3 float64, normalized to H[2, 2] == 1."""
    H = np.loadtxt(path)
    return H / H[2, 2]


def list_sequences(root=DEFAULT_ROOT, family=None):
    """All sequence dir names under root; optionally filter 'i_'/'v_'.

    Sorted alphabetically for deterministic demo/test runs.
    """
    names = sorted(d for d in os.listdir(root)
                   if os.path.isdir(os.path.join(root, d))
                   and d[:2] in ("i_", "v_"))
    if family:
        names = [n for n in names if n.startswith(family)]
    return names


def load_sequence(name, idx1=1, idx2=2, root=DEFAULT_ROOT, gray=True):
    """Load one image pair + GT homography from a sequence.

    Returns dict:
      name   -- sequence name
      family -- 'i' (illumination) or 'v' (viewpoint)
      im1    -- float64 grayscale of image idx1 (or None when gray=False
                in which case 'im1rgb' holds RGB uint8)
      im2    -- same for idx2
      im1rgb/im2rgb -- RGB uint8 (always present)
      H_1_k  -- 3x3 homography mapping image-1 coords -> image-k coords
                (identity when k == 1)
    """
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        raise FileNotFoundError("sequence %s not found under %s" % (name, root))
    seq = {
        "name": name,
        "family": name[0],
        "idx1": idx1,
        "idx2": idx2,
        "im1rgb": load_rgb_ppm(os.path.join(d, "%d.ppm" % idx1)),
        "im2rgb": load_rgb_ppm(os.path.join(d, "%d.ppm" % idx2)),
    }
    if gray:
        seq["im1"] = load_gray_ppm(os.path.join(d, "%d.ppm" % idx1))
        seq["im2"] = load_gray_ppm(os.path.join(d, "%d.ppm" % idx2))
    if idx1 == idx2:
        seq["H_1_k"] = np.eye(3)
    else:
        seq["H_1_k"] = load_homography(
            os.path.join(d, "H_%d_%d" % (idx1, idx2)))
    return seq


def gt_correspondences(seq, n=60, margin=16, seed=7):
    """Sample exact GT correspondences from the sequence's homography.

    Picks destination points uniformly in image 2 (inset by margin),
    maps them back through H_1_k^-1, keeps those whose source position
    is >= margin inside image 1. Same margin rule as
    transform_suite.gt_correspondences, so patches are fully supported
    on both sides even where the images differ in size.

    Returns (pts1, pts2): two (x, y)-order lists, exactly n entries.
    (Deterministic via seed; with a non-degenerate H every draw lands
    inside both insets unless the overlap itself is tiny.)
    """
    rng = np.random.RandomState(seed)
    H = seq["H_1_k"]
    h1, w1 = seq["im1"].shape if "im1" in seq else seq["im1rgb"].shape[:2]
    h2, w2 = seq["im2"].shape if "im2" in seq else seq["im2rgb"].shape[:2]
    Hinv = np.linalg.inv(H)
    pts1, pts2 = [], []
    tries = 0
    while len(pts2) < n and tries < 200 * n:
        tries += 1
        q = np.array([rng.uniform(margin, w2 - 1 - margin),
                      rng.uniform(margin, h2 - 1 - margin), 1.0])
        p = Hinv @ q
        if abs(p[2]) < 1e-9:
            continue
        px, py = p[0] / p[2], p[1] / p[2]
        if not (margin <= px < w1 - 1 - margin and margin <= py < h1 - 1 - margin):
            continue
        pts1.append((float(px), float(py)))
        pts2.append((float(q[0]), float(q[1])))
    if len(pts2) < n:
        raise ValueError("sequence %s: overlap too small for %d GT "
                         "correspondences at margin %d" % (seq["name"], n, margin))
    return pts1, pts2


def add_outliers(pts1, pts2, n_outliers=10, min_shift=60.0, seed=0x442):
    """Displace n_outliers destination points by >= min_shift px.

    Educational replacement for the old pts2.txt corruption: with 60 GT
    correspondences and 10 displaced, the inlier rate is 0.83 -- the
    same number RANSAC's iteration-count formula in Module 7 assumes.
    Returns (pts1, pts2_corrupt, outlier_idx).
    """
    rng = np.random.RandomState(seed)
    corrupt = list(pts2)
    idx = rng.choice(len(corrupt), size=min(n_outliers, len(corrupt)),
                     replace=False)
    for i in idx:
        ang = rng.uniform(0.0, 2.0 * np.pi)
        dist = min_shift + rng.uniform(0.0, 40.0)
        x, y = corrupt[i]
        corrupt[i] = (x + dist * np.cos(ang), y + dist * np.sin(ang))
    return pts1, corrupt, sorted(int(i) for i in idx)
