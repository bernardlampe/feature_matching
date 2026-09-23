"""BRIEF descriptor (Calonder et al. 2010) and dedicated tests set.

Binary descriptor: at each keypoint, compare intensity of smoothed
random pixel pairs yielding a n-bits bit string. Matching uses Hamming
distance. (No orientation compensation - that's ORB.)
"""

import numpy as np
from scipy import ndimage


def make_test_pattern(n_bits=256, patch=31, seed=0):
    """Random pairs (i, j) of pixel offsets in the patch, in canonical
    BRIEF style (Gaussian-distributed around center, iid)."""
    rng = np.random.RandomState(seed)
    half = patch // 2
    # coordinates drawn from a gaussian (sigma = patch/5), clamped
    pts = np.clip(np.round(rng.randn(n_bits * 2, 2) * patch / 5.0),
                  -half, half).astype(int)
    return [(tuple(pts[2 * i]), tuple(pts[2 * i + 1])) for i in range(n_bits)]


def brief_descriptors(gray, kps, n_bits=256, patch=31, seed=0,
                      pattern=None):
    """Returns (N, n_bits // 8) uint8 array of descriptors (one bit per
    test, packed) plus the tests used.

    Vectorized: all keypoints' patches are gathered at once with numpy
    advanced indexing; per-keypoint results are identical to the
    original per-keypoint reference.
    """
    if pattern is None:
        pattern = make_test_pattern(n_bits, patch, seed)
    img = ndimage.gaussian_filter(gray, 2.0)  # GII smoothing per paper
    half = patch // 2
    H, W = gray.shape
    Pi = np.array([p for p, _ in pattern])  # (n_bits, 2)
    Pj = np.array([q for _, q in pattern])
    kept_descs = []
    kept_kps = []
    # filter in-bounds keypoints first (same rule as the reference)
    ok = [(int(round(kp[0])), int(round(kp[1]))) for kp in kps
          if half <= int(round(kp[0])) < H - half
          and half <= int(round(kp[1])) < W - half]
    if not ok:
        return (np.zeros((0, n_bits // 8), dtype=np.uint8), kept_kps,
                pattern)
    ys = np.array([y for y, _ in ok])
    xs = np.array([x for _, x in ok])
    rows_i = ys[:, None] + Pi[None, :, 0]
    cols_i = xs[:, None] + Pi[None, :, 1]
    rows_j = ys[:, None] + Pj[None, :, 0]
    cols_j = xs[:, None] + Pj[None, :, 1]
    bits = img[rows_i, cols_i] < img[rows_j, cols_j]  # (N, n_bits)
    descs = np.packbits(bits, axis=1)
    kept_kps = [(y, x) for y, x in ok]
    return descs, kept_kps, pattern


def match_brief(d1, d2, max_dist=64, mutual=True):
    """Hamming NN matching on packed BRIEF bits. Returns (i, j, dist)."""
    # unpack to bits (N, n_bits)
    d01 = np.bitwise_xor(d1[:, None, :], d2[None, :, :])
    ham = np.zeros(d01.shape[:2], dtype=np.int32)
    # popcount lookup (fast enough for course-size sets)
    POP = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
    for k in range(d01.shape[2]):
        ham += POP[d01[:, :, k]]
    best = np.argmin(ham, axis=1)
    matches = []
    for i, j in enumerate(best):
        ham_i = ham[i].copy()
        ham_i[j] = np.iinfo(np.int32).max
        second = ham_i.min()
        h = int(ham[i, j])
        # ratio test with the zero-distance hole closed: when h == 0 AND
        # a second 0-distance candidate exists, the pair is fully
        # ambiguous and must be REJECTED (the old `second == 0` accept
        # branch let duplicates through, defeating the ratio test)
        if h <= max_dist and not (h == 0 and second == 0) and \
                (second == 0 or h / max(second, 1) < 0.8):
            matches.append((i, j, h))
    if mutual:
        # true cross-check: reverse NN over the FULL Hamming matrix
        rev_nn = np.argmin(ham, axis=0)
        matches = [(i, j, d) for i, j, d in matches if rev_nn[j] == i]
    return matches
