"""Combine method outputs without averaging or distance-based suppression."""

import pandas as pd


def combine_candidates(grid, voronoi):
    """Keep every input row, its method and its original candidate ID.

    The existing symmetry function removes equivalent positions afterwards.
    Grid rows come first, followed by Voronoi rows.
    """
    combined = pd.concat(
        [
            grid.rename(columns={"candidate_id": "source_candidate_id"}).assign(
                source_method="grid"
            ),
            voronoi.rename(columns={"candidate_id": "source_candidate_id"}).assign(
                source_method="voronoi"
            ),
        ],
        ignore_index=True,
    )
    combined.insert(0, "candidate_id", range(len(combined)))
    return combined
