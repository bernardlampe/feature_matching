"""MSER: Maximally Stable Extremal Regions (Matas et al. 2002).

Component tree over increasing intensity thresholds via union-find,
run on BOTH polarities (gray and its inverse). With `gray <= t` growth
only, bright plateaus nested in darker surrounds are absorbed into the
background component the moment they appear and can never be stable —
the canonical negative test (nested plateaus 0/100/200) returns zero
regions without the inverted pass. Dark-on-light is the `<=` pass;
light-on-dark is the inverted pass.

Region identity = first-activated pixel, carried by the surviving
(larger) root through unions; a region's trajectory records its area
at each threshold level while it exists (it ends when absorbed).

Stability per Matas eq. 3: for a region with trajectory snapshots
(t_i, s_i), variation = min_i |s_j - s_i| / s_i with t_j the region's
size measured at most `delta` intensity levels after t_i. Maximally
stable when that minimum is below `max_variation`.
"""

import numpy as np
from scipy import ndimage


def _trajectory(flat, H, W, n_px, delta):
    """Component-tree walk over one polarity. Returns
    {identity_pixel: [(thr, size), ...]} trajectories."""
    order = np.argsort(flat, kind="stable")
    parent = list(range(n_px))
    area = [1] * n_px
    identity = list(range(n_px))
    active = np.zeros(n_px, dtype=bool)
    history = {}
    live_roots = set()

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    k = 0
    while k < n_px:
        thr = float(flat[order[k]])
        # activate every pixel at this intensity level at once
        j = k
        new_idx = []
        while j < n_px and flat[order[j]] == thr:
            idx = int(order[j])
            active[idx] = True
            new_idx.append(idx)
            live_roots.add(idx)
            j += 1
        # union newly active pixels with active 4-neighbors
        survivors = set()
        absorbed = set()
        for idx in new_idx:
            y0, x0 = divmod(idx, W)
            touched = {find(idx)}
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y0 + dy, x0 + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                nidx = ny * W + nx
                if active[nidx]:
                    touched.add(find(nidx))
            if len(touched) > 1:
                cand = sorted(touched, key=lambda r: (-area[r], r))
                survivor = cand[0]
                for r in cand[1:]:
                    parent[r] = survivor
                    area[survivor] += area[r]
                    if r in live_roots:
                        live_roots.discard(r)
                        absorbed.add(r)
                live_roots.add(survivor)
                survivors.add(survivor)
            else:
                survivors.update(touched)
        # snapshot: every live region at this level. New singletons born
        # here and previously-live regions both record (thr, size);
        # absorbed regions no longer receive snapshots (they ceased to
        # exist as separate regions).
        for r in live_roots:
            history.setdefault(identity[r], []).append((thr, area[r]))
        k = j
    return history


def _stable_regions(hist, n_px, delta, min_area, max_area, max_variation):
    """Matas stability over trajectories -> [(identity, (thr, size))].

    A component-tree node's pixel set is CONSTANT after birth; its
    trajectory entries are (level, area_of_component_at_that_level).
    The area at t_i + delta is looked up as the last entry with level
    <= t_i + delta (carried area) via one vectorized searchsorted per
    trajectory — the scalar nested-loop variant spent ~50M searchsorted
    calls (minutes) on the real pair per the post-fix audit profile.
    """
    stables = []
    for fp, snaps in hist.items():
        s = np.asarray([s_ for _, s_ in snaps], dtype=np.float64)
        t = np.asarray([t_ for t_, _ in snaps], dtype=np.float64)
        order = np.argsort(t, kind="stable")
        s, t = s[order], t[order]
        ok = s <= 0.5 * n_px
        if not ok.any():
            continue
        # carried area at each t_i + delta, ONE searchsorted call
        j = np.searchsorted(t, t + delta, side="right") - 1
        valid = ok & (j >= np.arange(len(s)))
        var = np.abs(s[np.maximum(j, 0)] - s) / np.maximum(s, 1.0)
        var[~valid] = np.inf
        k = int(np.argmin(var))
        if not np.isfinite(var[k]):
            continue
        best_var, best_snap = var[k], (t[k], s[k])
        if best_var < max_variation and min_area <= best_snap[1] <= max_area:
            stables.append((fp, best_snap))
    return stables


def mser_regions(gray, delta=5, min_area=60, max_area=8000,
                 max_variation=0.25):
    """Returns list of (centroid_y, centroid_x, area) in the ORIGINAL
    gray frame."""
    gray = np.asarray(gray, dtype=np.float64)
    H, W = gray.shape
    n_px = H * W
    out = []
    for polarity in (1.0, -1.0):
        img = gray if polarity > 0 else 255.0 - gray
        hist = _trajectory(np.asarray(img).ravel(), H, W, n_px, delta)
        stables = _stable_regions(hist, n_px, delta, min_area, max_area,
                                  max_variation)
        # recover pixels by flood-fill at the recorded threshold.
        # ndimage.label on a 512x384 mask costs ~10 ms; sharing one
        # labeled threshold across all candidates at that threshold
        # (distinct thresholds are few) keeps this stage linear.
        labels_at_thr = {}
        for fp, (thr, sz) in stables[:200]:  # cap for demo speed
            if thr not in labels_at_thr:
                mask = np.asarray(img) <= thr
                labels_at_thr[thr] = (ndimage.label(mask), mask)
            (lbl, _), mask = labels_at_thr[thr]
            ry, rx = divmod(fp, W)
            if not (0 <= ry < H and 0 <= rx < W) or not mask[ry, rx]:
                continue
            comp = lbl[ry, rx]
            ys, xs = np.nonzero(lbl == comp)
            if min_area <= len(ys) <= max_area:
                out.append((ys.mean(), xs.mean(), len(ys)))
    # dedupe near-identical centroids
    dedup = []
    for cy, cx, a in out:
        if all((cy - d[0]) ** 2 + (cx - d[1]) ** 2 > 100 for d in dedup):
            dedup.append((cy, cx, a))
    return dedup
