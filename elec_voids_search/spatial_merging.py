"""Periodic distance-only grouping around ranked representatives, no symmetry."""
import math
import numpy as np


def representative_order(table, representative='largest_center_distance'):
    if representative == 'largest_center_distance':
        return sorted(range(len(table)), key=lambda i: (-float(table.iloc[i].nearest_atom_center_A), i))
    if representative in {'grid', 'voronoi'}:
        return sorted(range(len(table)), key=lambda i: (table.iloc[i].source_method != representative, i))
    if representative == 'first':
        return list(range(len(table)))
    raise ValueError('Unknown representative policy')


def representative_ball_groups(structure, positions, distance, priority):
    """Every absorbed member is directly closer than distance to its leader.

    Rank leaders first, then suppress only their direct periodic neighbors.
    No transitive chains: an absorbed point cannot pull in a distant point.
    Leaders remain pairwise separated by at least distance.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if distance is None:
        return [[i] for i in range(len(positions))]
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError('distance must be None or finite and positive')
    remaining = np.ones(len(positions), dtype=bool)
    groups = []
    for chosen in priority:
        if not remaining[chosen]:
            continue
        indices = np.flatnonzero(remaining)
        distances = structure.lattice.get_all_distances([positions[chosen]], positions[indices])[0]
        members = indices[distances < distance].tolist()
        groups.append(members)
        remaining[members] = False
    return groups

