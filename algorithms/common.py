"""Shared utilities: grayscale conversion and Gaussian filtering."""

import math

import numpy as np
from PIL import Image
from scipy import ndimage

def load_gray(fname):
    im = np.asarray(Image.open(fname).convert("RGB"))
    # MATLAB rgb2gray weights
    g = 0.2989 * im[..., 0] + 0.5870 * im[..., 1] + 0.1140 * im[..., 2]
    return g


def gaussian_kernel1d(sigma):
    half = max(1, int(math.ceil(2.5 * sigma)))
    x = np.arange(-half, half + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(img, sigma):
    k = gaussian_kernel1d(sigma)
    v = ndimage.convolve1d(img, k, axis=0, mode="nearest")
    return ndimage.convolve1d(v, k, axis=1, mode="nearest")
