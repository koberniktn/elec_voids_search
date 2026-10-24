import numpy as np
from .config import DEFAULT_CONFIG


def pbc_distance(structure, frac_a, frac_b):
    distance, _ = structure.lattice.get_distance_and_image(frac_a, frac_b)
    return float(distance)


def symmetry_equivalent(structure, point_a, point_b, symmetry_operations, SYMMETRY_TOLERANCE=DEFAULT_CONFIG.symmetry_tolerance_A):
    for operation in symmetry_operations:
        transformed = np.mod(operation.operate(point_a), 1.0)
        if pbc_distance(structure, transformed, point_b) < SYMMETRY_TOLERANCE:
            return True
    return False


def symmetry_nonequivalent_voids(structure, cluster_frac_positions, analyzer, SYMMETRY_TOLERANCE=DEFAULT_CONFIG.symmetry_tolerance_A, SYMPREC=DEFAULT_CONFIG.symmetry_symprec_A, return_groups=False):
    
    symmetry_operations = analyzer.get_symmetry_operations(cartesian=False)

    parent = list(range(len(cluster_frac_positions)))

    def find_root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(cluster_frac_positions)):
        for j in range(i + 1, len(cluster_frac_positions)):
            if symmetry_equivalent(
                structure,
                cluster_frac_positions[i],
                cluster_frac_positions[j],
                symmetry_operations,
                SYMMETRY_TOLERANCE
            ):
                root_i, root_j = find_root(i), find_root(j)
                if root_i != root_j:
                    parent[root_j] = root_i

    symmetry_groups = {}
    for i in range(len(cluster_frac_positions)):
        symmetry_groups.setdefault(find_root(i), []).append(i)

    if return_groups:
        return list(symmetry_groups.values())
    return [cluster_frac_positions[indices[0]] for indices in symmetry_groups.values()]
