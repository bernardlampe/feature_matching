A review of feature matching, built around the [HPatches](https://hpatches.github.io) sequences dataset

```sh
python3 -m pip install -r requirements.txt
python3 stitch_tool.py pano.png img_a.png img_b.png img_c.png
```

## Review includes:
    — Local feature detection: corners
    — Fast corners
    — Scale-space and covariant regions
    — Descriptors
    — Matching
    — Template tracking (matchless correspondence)
    — Robust model estimation
    — Alternative geometry: epipolar constraint
    — Application chaser: image stitching
    — Matching at scale: index structures
    — Two cameras: essential matrix and relative pose
