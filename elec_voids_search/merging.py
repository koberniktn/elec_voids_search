"""Merge close positions only in the combined grid/Voronoi candidate list."""

import pandas as pd
import math

from .config import DEFAULT_CONFIG
from .periodic_groups import merge_close_vertices_batched
from .spatial_merging import representative_ball_groups, representative_order


def merge_candidates(structure, combined,
                     distance=DEFAULT_CONFIG.merge_distance_A,
                     representative=DEFAULT_CONFIG.merge_representative,
                     grouping=DEFAULT_CONFIG.merge_grouping):
    """Group by the original connected-component rule by default.

    Return the annotated complete input, selected representatives and a cluster
    table. All original IDs and coordinates remain available. None disables
    merging. Coordinates are never averaged or optimized.
    """
    table = combined.copy().reset_index(drop=True)
    if not table.candidate_id.is_unique:
        raise ValueError("Combined candidate IDs must be unique")
    if representative not in {"first", "grid", "voronoi", "largest_center_distance"}:
        raise ValueError("Unknown representative policy")
    if grouping not in {"representative", "connected"}:
        raise ValueError("Unknown grouping policy")
    if distance is not None and (not math.isfinite(distance) or distance <= 0):
        raise ValueError("distance must be positive or None")
    positions = table[["frac_x", "frac_y", "frac_z"]].to_numpy()
    if distance is None or not len(table):
        groups = [[i] for i in range(len(table))]
    elif grouping == "representative":
        groups = representative_ball_groups(structure, positions, distance,
                                            representative_order(table, representative))
    else:
        points, _ = merge_close_vertices_batched(structure, positions, MERGE_DISTANCE=distance)
        groups = [item["voronoi_members"] for item in points]

    for column in ["merge_cluster_id", "merge_member_count", "merge_representative_candidate_id"]:
        table[column] = pd.Series([-1] * len(table), dtype="int64")
    for column in ["merge_member_candidate_ids", "merge_source_methods"]:
        table[column] = pd.Series([None] * len(table), dtype="object")
    table["is_merge_representative"] = False
    selected_rows, clusters = [], []
    for cluster_id, members in enumerate(groups):
        chosen = members[0]
        if representative in {"grid", "voronoi"}:
            preferred = [i for i in members if table.at[i, "source_method"] == representative]
            if preferred:
                chosen = preferred[0]
        elif representative == "largest_center_distance":
            chosen = table.loc[members, "nearest_atom_center_A"].idxmax()
        candidate_id = int(table.at[chosen, "candidate_id"])
        member_ids = table.loc[members, "candidate_id"].tolist()
        methods = sorted(set(table.loc[members, "source_method"]))
        table.loc[members, "merge_cluster_id"] = cluster_id
        table.loc[members, "merge_member_count"] = len(members)
        table.loc[members, "merge_representative_candidate_id"] = candidate_id
        for member in members:
            table.at[member, "merge_member_candidate_ids"] = member_ids
            table.at[member, "merge_source_methods"] = methods
        table.at[chosen, "is_merge_representative"] = True
        selected_rows.append(chosen)
        span = float(structure.lattice.get_all_distances(
            [positions[chosen]], positions[members]
        ).max())
        clusters.append({
            "merge_cluster_id": cluster_id,
            "representative_candidate_id": candidate_id,
            "member_candidate_ids": member_ids,
            "member_source_methods": methods,
            "member_count": len(members),
            "representative_frac_coords": positions[chosen].tolist(),
            "max_distance_to_representative_A": span,
        })
    columns = ["merge_cluster_id", "representative_candidate_id", "member_candidate_ids",
               "member_source_methods", "member_count", "representative_frac_coords",
               "max_distance_to_representative_A"]
    return table, table.iloc[selected_rows].copy(), pd.DataFrame(clusters, columns=columns)
