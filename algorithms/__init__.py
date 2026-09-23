"""Reference implementations for the feature-matching course.

Pure numpy/scipy implementations, no OpenCV. Modules are grouped by
course topic:

- detectors: harris_detector, fast_detector, dog_detector, mser_detector
- descriptors: descriptors (patch, SIFT-like), brief_descriptor
- matching: matching (NN + ratio/cross-check), ann_matcher (k-d tree),
  ncc_matcher (template search), klt_tracker (Lucas-Kanade)
- geometry/robust estimation: ransac (homography), fundamental,
  essential, stitcher
- utilities: common (image/color helpers), visualize,
  transform_suite (synthetic transforms + GT correspondences)
- legacy CLI: harris (MATLAB-ported standalone harris demo)

Import as e.g. `from algorithms import ransac` (course_demo/tests do
this via sys.path manipulation), or run the package demo from the repo
root with `python3 course_demo.py [outdir]`.
"""
