import numpy as np
from itertools import product
from scipy import ndimage
from scipy.spatial import cKDTree
 
 
def find_local_minima(distribution, size=3):
    E = distribution[:-1, :-1, :-1]
    minf = ndimage.minimum_filter(E, size=size, mode='wrap')
    mask = E == minf
 
    labeled, count = ndimage.label(mask, structure=np.ones((3, 3, 3), bool))
    # Join connected plateaus across periodic faces, including edges/corners.
    # Padding alone does not join labels split across opposite cell boundaries.
    parent = np.arange(count + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for axis in range(3):
        low = np.take(labeled, 0, axis=axis)
        high = np.take(labeled, -1, axis=axis)
        for shift in product((-1, 0, 1), repeat=2):
            opposite = np.roll(high, shift, axis=(0, 1))
            connected = (low > 0) & (opposite > 0) & (low != opposite)
            for a, b in np.unique(np.column_stack((low[connected], opposite[connected])), axis=0):
                ra, rb = root(a), root(b)
                if ra != rb:
                    parent[rb] = ra

    roots = np.array([root(i) for i in range(count + 1)])
    flat = np.flatnonzero(mask)
    _, first = np.unique(roots[labeled.ravel()[flat]], return_index=True)
    idx = np.array(np.unravel_index(flat[np.sort(first)], E.shape)).T
    val = E[tuple(idx.T)]
    return idx, val
 
 
def cluster_minima(idx, val, shape, cell, min_distance, max_value=None):
    """Optional legacy distance suppression; not used by default for candidates."""
    if max_value is not None:
        m = val < max_value
        idx, val = idx[m], val[m]

    if not len(idx):
        return []
 
    cell = np.asarray(cell, dtype=float)
    frac = idx / np.array(shape, dtype=float)
    cart = frac @ cell
 
    shifts = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij')).reshape(3, -1).T
    n = len(idx)
    rep_cart = ((shifts[:, None, :] + frac[None, :, :]).reshape(-1, 3)) @ cell
    owner = np.tile(np.arange(n), len(shifts))
    tree = cKDTree(rep_cart)
 
    order = np.argsort(val)
    alive = np.ones(n, bool)
 
    clusters = []
    for i in order:
        if not alive[i]:
            continue
        neighbors = np.unique(owner[tree.query_ball_point(cart[i], min_distance)])
        clusters.append((idx[i], val[i], len(neighbors)))
        for j in neighbors:
            alive[j] = False
 
    return clusters


def clustered_local_minima(data, min_distance=None, relative_max_value=None, cell=None, size=3):
    """Return all periodic local minima, one representative per connected plateau.

    Distance suppression and a cutoff relative to the global minimum are opt-in:
    both can remove genuine interstitial candidates before any ion-size check.
    Existing callers can still request the old behaviour explicitly.
    """
    idx, val = find_local_minima(data, size=size)
    if min_distance is None:
        if relative_max_value is not None:
            keep = val < data.min() + relative_max_value
            idx, val = idx[keep], val[keep]
        return [(i, v, 1) for i, v in zip(idx, val)]
    if cell is None:
        raise ValueError("cell is required for distance suppression")
    core_shape = tuple(s - 1 for s in data.shape)
    clusters = cluster_minima(idx, val, core_shape,
                              min_distance=min_distance,
                              max_value=None if relative_max_value is None else data.min() + relative_max_value,
                              cell = cell
                             )
    return clusters
