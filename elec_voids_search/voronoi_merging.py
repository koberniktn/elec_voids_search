import numpy as np
import pandas as pd
from .config import DEFAULT_CONFIG


def merge_close_voronoi_vertices(structure, raw_vertices, MERGE_DISTANCE=DEFAULT_CONFIG.merge_distance_A):
    def pbc_distance(frac_a, frac_b):
        distance, _ = structure.lattice.get_distance_and_image(frac_a, frac_b)
        return float(distance)


    parent = list(range(len(raw_vertices)))  # индекс элемента =  индекс атома, значение = индекс родителя (репрезентирующего атома кластера)

    def find_root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i


    def union(i, j):
        root_i, root_j = find_root(i), find_root(j)
        if root_i != root_j:
            parent[root_j] = root_i


    for i in range(len(raw_vertices)):
        for j in range(i + 1, len(raw_vertices)):
            if pbc_distance(raw_vertices[i], raw_vertices[j]) < MERGE_DISTANCE:
                union(i, j)

    clusters = {}
    for i in range(len(raw_vertices)):
        clusters.setdefault(find_root(i), []).append(i)

    merged_points = []
    for cluster_id, members in enumerate(clusters.values()):
        merged_points.append({
            "cluster_id": cluster_id,
            "frac": raw_vertices[members[0]],
            "voronoi_members": members,
            "merged_vertex_count": len(members),
        })

    merged_df = pd.DataFrame([{
        "cluster_id": item["cluster_id"],
        "frac_x": item["frac"][0],
        "frac_y": item["frac"][1],
        "frac_z": item["frac"][2],
        "merged_vertex_count": item["merged_vertex_count"],
    } for item in merged_points])

    return merged_points, merged_df
