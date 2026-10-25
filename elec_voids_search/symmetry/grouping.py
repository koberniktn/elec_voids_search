import numpy as np


def component_labels(n, pairs):
    parent = np.arange(n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    roots = [root(i) for i in range(n)]
    mapping = {}
    return np.array([mapping.setdefault(r, len(mapping)) for r in roots], dtype=int)
