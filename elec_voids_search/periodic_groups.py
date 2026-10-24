"""Batched exact lattice distances; same strict thresholds and graph components.

The original notebook implementations remain available in symmetry.py and
voronoi_merging.py. This module changes evaluation cost, not grouping criteria.
"""
import numpy as np
import pandas as pd
from .grouping import component_labels


DISTANCE_BATCH_SIZE = 128


def periodic_groups(structure, positions, tolerance, operations=None):
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(positions)
    connected = np.zeros((n, n), dtype=bool)
    transformed_sets = (positions,) if operations is None else (
        np.mod(positions @ op.rotation_matrix.T + op.translation_vector, 1.)
        for op in operations
    )
    for transformed in transformed_sets:
        for start in range(0, n, DISTANCE_BATCH_SIZE):
            end = min(start + DISTANCE_BATCH_SIZE, n)
            distances = structure.lattice.get_all_distances(transformed[start:end], positions)
            connected[start:end] |= distances < tolerance
    pairs = np.argwhere(np.triu(connected, k=1))
    labels = component_labels(n, pairs)
    return [np.flatnonzero(labels == label).tolist() for label in range(len(set(labels)))]


def merge_close_vertices_batched(structure, raw_vertices, MERGE_DISTANCE):
    groups = periodic_groups(structure, raw_vertices, MERGE_DISTANCE)
    points = [dict(cluster_id=group_id, frac=raw_vertices[members[0]],
                   voronoi_members=members, merged_vertex_count=len(members))
              for group_id, members in enumerate(groups)]
    rows = [dict(cluster_id=item['cluster_id'], frac_x=item['frac'][0],
                 frac_y=item['frac'][1], frac_z=item['frac'][2],
                 merged_vertex_count=item['merged_vertex_count']) for item in points]
    return points, pd.DataFrame(rows)
