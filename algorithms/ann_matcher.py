"""Approximate nearest-neighbor matching via k-d tree (Module 10).

The course's `matching.py` scans every descriptor pair: O(N1*N2). Real
systems match tens of thousands of keypoints, so NN search goes through
a spatial index. This module implements a small median k-d tree over
real-valued descriptors (SIFT-like 128-D) with bounded-backtracking
exact search (Friedman/Bentley/Finkel 1977 style; note this is NOT a
BBD-tree and NOT FLANN's randomized-forest — those are cited as
background in docs/module10_ann.md), plus brute-force reference code. For
binary descriptors the honest baseline is one vectorized Hamming pass
(`query_brief_bruteforce`), which is what OpenCV's BFMatcher does at
course scale; multi-index hashing is exercise 10.2.
"""

import heapq

import numpy as np


# ----------------------------------------------------------------- k-d tree

class _KDNode:
    __slots__ = ("idx", "split", "left", "right")

    def __init__(self, idx, split, left, right):
        self.idx = idx      # index of the point held at this node
        self.split = split  # axis to compare against (-1 for leaf)
        self.left = left
        self.right = right


def _build(points, idxs, depth):
    if len(idxs) == 0:
        return None
    axis = depth % points.shape[1]
    order = np.argsort(points[idxs, axis], kind="stable")
    idxs = np.asarray(idxs)[order]
    mid = len(idxs) // 2
    return _KDNode(
        int(idxs[mid]), axis,
        _build(points, idxs[:mid], depth + 1),
        _build(points, idxs[mid + 1:], depth + 1))


class AnnMatcher:
    """k-d tree NN matcher over unit-norm real descriptors.

    `query` returns for each source descriptor the tree's (exact, up to
    the bounded backtracking) nearest neighbor index and L2 distance.
    The `exact` flag and `leaf_size` are accepted for API compatibility
    but currently unused: the tree splits to single points and the
    backtracking scan is exact. Making leaf-stopping + early termination
    work (a true approximate mode) is exercise 10.1 in docs/module10_ann.md.
    """

    def __init__(self, data, leaf_size=16):
        data = np.asarray(data, dtype=np.float64)
        self.data = data
        self.root = _build(data, np.arange(len(data)), 0)
        self.leaf_size = leaf_size

    def query(self, queries, exact=False, return_best2=False):
        queries = np.asarray(queries, dtype=np.float64)
        out_j = np.empty(len(queries), dtype=np.int64)
        out_d = np.empty(len(queries))
        best2_d = np.full(len(queries), np.inf) if return_best2 else None
        for qi, q in enumerate(queries):
            heap = [(0.0, 0)]  # (dist to node bbox, node) — simulated
            best = (np.inf, -1)
            second = np.inf
            stack = [(self.root, 0.0)] if self.root else []
            while stack:
                node, bound = stack.pop()
                if bound >= best[0]:
                    continue
                d = np.linalg.norm(self.data[node.idx] - q)
                if d < best[0]:
                    second, best = best[0], (d, node.idx)
                elif d < second:
                    second = d
                diff = q[node.split] - self.data[node.idx, node.split]
                near, far = ((node.left, node.right) if diff < 0
                             else (node.right, node.left))
                if near is not None:
                    stack.append((near, bound))
                if far is not None:
                    # |diff| is a valid lower bound on any point on the far
                    # side (squared distance along the split axis)
                    stack.append((far, max(bound, abs(diff))))
            out_j[qi], out_d[qi] = best[1], best[0]
            if return_best2:
                best2_d[qi] = second
        if return_best2:
            return _AnnResult(out_j, out_d, best2_d)
        return _AnnResult(out_j, out_d)


class _AnnResult:
    """Lightweight namespace: .j (NN indices), .d (distances),
    .d2 (second-best distances)."""

    def __init__(self, j, d, d2=None):
        self.j = j
        self.d = d
        self.d2 = d2


def brute_force_match(d1, d2):
    """Reference O(N1*N2) NN: the baseline the course's matching.py uses.
    Returns _AnnResult for parity with AnnMatcher.query."""
    dd = 2.0 - 2.0 * (np.asarray(d1) @ np.asarray(d2).T)
    j = np.argmin(dd, axis=1)
    return _AnnResult(j, np.sqrt(np.maximum(dd[np.arange(len(d1)), j], 0)))


# ------------------------------------------------------------ BRIEF binary

def query_brief_bruteforce(d1, d2, max_dist=64):
    """Vectorized Hamming brute force over packed uint8 descriptors —
    the honest baseline for binary descriptors (single XOR-popcount pass).
    Returns (j, hamming) arrays."""
    d1 = np.asarray(d1, dtype=np.uint8)
    d2 = np.asarray(d2, dtype=np.uint8)
    POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    xor = np.bitwise_xor(d1[:, None, :], d2[None, :, :])
    ham = POP[xor].sum(axis=2)
    j = np.argmin(ham, axis=1)
    ham_j = ham[np.arange(len(d1)), j]
    j[ham_j > max_dist] = -1
    return j, ham_j
