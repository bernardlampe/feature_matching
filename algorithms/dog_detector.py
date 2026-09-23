"""DoG scale-space keypoint detector (SIFT section 4.1 core).

Extrema over scale and space in a Difference-of-Gaussians pyramid with
contrast rejection and edge (principal-curvature ratio) rejection.
"""

import math

import numpy as np
from scipy import ndimage

from .common import gaussian_blur


def dog_keypoints(gray, num_octaves=3, scales_per_octave=3, sigma0=1.6,
                  contrast_thresh=0.03, edge_ratio=10.0):
    """Returns list of (y, x, sigma)."""
    k = 2.0 ** (1.0 / scales_per_octave)
    kps = []
    # load_gray returns [0, 255]; SIFT's contrast_thresh=0.03 is defined
    # on [0,1] images — scale it or the gate passes ~82% of pixels
    contrast_cut = contrast_thresh * 255.0
    for oct_idx in range(num_octaves):
        im = gray if oct_idx == 0 else ndimage.zoom(
            gray, 1.0 / 2.0 ** oct_idx, order=1)
        # SIFT §4.1: the base of the pyramid sits at sigma0/4, not at
        # raw intensity; without this pre-blur the first DoG layer
        # (d=1) is nearly empty (1 extremum on the first image vs ~100
        # at d=2,3)
        im = gaussian_blur(im, sigma0 / 4.0)
        scales = [im]
        sig = sigma0 / 4.0
        for s in range(scales_per_octave + 2):
            sig_next = sigma0 * k ** (s + 1)
            need = math.sqrt(max(sig_next ** 2 - sig ** 2, 1e-6))
            scales.append(gaussian_blur(scales[-1], need))
            sig = sig_next
        dogs = [scales[i + 1] - scales[i] for i in range(len(scales) - 1)]
        H, W = dogs[0].shape
        for d in range(1, len(dogs) - 1):
            D = dogs[d]
            up, dn = dogs[d - 1], dogs[d + 1]
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    v = D[y, x]
                    if abs(v) < contrast_cut:
                        continue
                    block = np.concatenate([
                        up[y - 1:y + 2, x - 1:x + 2].ravel(),
                        D[y - 1:y + 2, x - 1:x + 2].ravel(),
                        dn[y - 1:y + 2, x - 1:x + 2].ravel()])
                    block = np.delete(block, 13)  # drop center == v
                    if (v > 0 and v >= block.max()) or \
                       (v < 0 and v <= block.min()):
                        # edge rejection: 2x2 Hessian principal curvatures
                        dxx = D[y, x - 1] + D[y, x + 1] - 2 * v
                        dyy = D[y - 1, x] + D[y + 1, x] - 2 * v
                        dxy = (D[y + 1, x + 1] - D[y + 1, x - 1] -
                               D[y - 1, x + 1] + D[y - 1, x - 1]) / 4.0
                        tr = dxx + dyy
                        det = dxx * dyy - dxy * dxy
                        if det <= 0:
                            continue
                        if tr * tr / det < (edge_ratio + 1) ** 2 / edge_ratio:
                            # upper octaves are detected in the ZOOMED frame;
                            # resample back to full-res coordinates or every
                            # consumer (descriptors, demos, tests) silently
                            # reads the wrong image content
                            scale_f = 2.0 ** oct_idx
                            scale = sigma0 * k ** (d) * scale_f
                            kps.append((y * scale_f, x * scale_f, scale))
    return kps
