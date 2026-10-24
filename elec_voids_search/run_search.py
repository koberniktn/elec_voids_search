"""Notebook API: generate, combine, merge close candidates, symmetry, then geometry."""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .combination import combine_candidates
from .config import resolve_config
from .exports import export_candidates
from .grid_search import grid_candidates
from .geometry import screen_geometry
from .ionic_geometry import validate_oxidation_states
from .merging import merge_candidates
from .periodic_groups import periodic_groups
from .voronoi_search import voronoi_candidates


def run_search(structure_path, output_dir, config=None, save_results=True, **overrides):
    """Run candidate generation and geometry screening with SearchConfig or explicit setting overrides.

    No distance merging is applied inside either generator. The complete lists
    are combined first; close-site merging precedes symmetry selection. Ionic
    exclusion models are audited. Resolved settings are saved with every result.
    save_results=False returns tables in memory without writing output files.
    """
    settings = resolve_config(config, **overrides)
    structure_path = Path(structure_path).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    structure = Structure.from_file(structure_path)
    validate_oxidation_states(structure, settings)

    # 1. All grid maxima and all Voronoi vertices, independently.
    grid, grid_parameters = grid_candidates(structure_path, config=settings)
    voronoi, voronoi_parameters, voronoi_details = voronoi_candidates(
        structure, config=settings, return_details=True
    )

    # 2. One complete candidate list; no coordinates are discarded here.
    combined = combine_candidates(grid, voronoi)

    # 3. The same close-site grouping acts on candidates from BOTH methods.
    combined, merged, clusters = merge_candidates(
        structure, combined, distance=settings.merge_distance_A,
        representative=settings.merge_representative, grouping=settings.merge_grouping,
    )

    # 4. Symmetry reduction happens before any probe-radius screening.
    analyzer = SpacegroupAnalyzer(
        structure, symprec=settings.symmetry_symprec_A,
        angle_tolerance=settings.symmetry_angle_tolerance_deg,
    )
    positions = merged[["frac_x", "frac_y", "frac_z"]].to_numpy()
    groups = periodic_groups(
        structure, positions, settings.symmetry_tolerance_A,
        operations=analyzer.get_symmetry_operations(cartesian=False),
    )
    merged["symmetry_group_id"] = pd.Series(index=merged.index, dtype="int64")
    for group_id, indices in enumerate(groups):
        merged.loc[merged.index[indices], "symmetry_group_id"] = group_id
    merged["symmetry_group_id"] = merged.symmetry_group_id.astype(int)
    group_by_cluster = merged.set_index("merge_cluster_id").symmetry_group_id
    combined["symmetry_group_id"] = combined.merge_cluster_id.map(group_by_cluster).astype(int)
    representative_rows = [
        merged.index[indices[0]] for indices in groups
    ]
    nonequivalent = merged.loc[representative_rows].copy()
    selected_ids = set(nonequivalent.candidate_id)
    for table in [combined, merged, nonequivalent]:
        table["is_symmetry_representative"] = table.candidate_id.isin(selected_ids)

    # 5. Evaluate only the representatives selected before geometry.
    geometry_results = screen_geometry(structure, nonequivalent, settings)

    summary = pd.DataFrame([{
        "structure_file": structure_path.name,
        "formula": structure.composition.reduced_formula,
        "space_group": analyzer.get_space_group_symbol(),
        "n_grid_candidates": len(grid),
        "n_voronoi_raw_vertices": len(voronoi),
        "n_voronoi_candidates": len(voronoi),
        "n_combined_candidates": len(combined),
        "n_merged_candidates": len(merged),
        "n_symmetry_nonequivalent_candidates": len(nonequivalent),
        "n_geometry_passed": len(geometry_results["geometry"]),
        "n_geometry_indeterminate": len(geometry_results["geometry_indeterminate"]),
        "n_geometry_rejected": len(geometry_results["geometry_rejected"]),
        "n_geometry_retained": len(geometry_results["geometry_retained"]),
        "n_full_radius_passed": int((geometry_results["geometry_evaluated"].full_radius_status == "passed").sum()),
        "probe_element": settings.probe_element,
        "probe_oxidation_state": settings.probe_oxidation_state,
        "geometry_minimum_fit_fraction": settings.geometry_minimum_fit_fraction,
        "grid_resolution_A": settings.grid_resolution_A,
        "merge_distance_A": settings.merge_distance_A,
        "merge_representative": settings.merge_representative,
        "merge_grouping": settings.merge_grouping,
        "symprec_A": settings.symmetry_symprec_A,
        "symmetry_tolerance_A": settings.symmetry_tolerance_A,
    }])
    parameters = {
        "settings": settings.to_dict(),
        "grid": grid_parameters,
        "voronoi": voronoi_parameters,
        "stage_order": ["grid_and_voronoi", "combined", "merged", "nonequivalent", "geometry"],
        "combination_order": ["grid", "voronoi"],
        "radius_filter_applied": True,
        "periodic_grouping": "exact_periodic_distances",
        "pre_geometry_merge_grouping": settings.merge_grouping,
        "geometry_coordinates": "fixed_preselected_symmetry_representatives",
        "symmetry_reduction_stage": "before_geometry_only",
    }
    if save_results:
        output_dir.mkdir(parents=True, exist_ok=False)
        for name, table in [
            ("grid", grid), ("voronoi", voronoi), ("combined", combined),
            ("merged", merged), ("nonequivalent", nonequivalent),
            *[(name, geometry_results[name]) for name in ["geometry_evaluated", "geometry",
               "geometry_indeterminate", "geometry_rejected", "geometry_retained"]],
        ]:
            export_candidates(structure, table, output_dir, name, parameters, write_cif=settings.export_cif)
        voronoi_details["raw_voronoi_df"].to_csv(output_dir / "raw_voronoi_vertices.csv", index=False)
        clusters.to_csv(output_dir / "candidate_clusters.csv", index=False)
        (output_dir / "candidate_clusters.json").write_text(
            clusters.to_json(orient="records", indent=2, force_ascii=False, double_precision=15) + "\n"
        )
        for name in ["geometry_candidate_audit", "site_radii", "occupied_probe_controls"]:
            table = geometry_results[name]
            table.to_csv(output_dir / f"{name}.csv", index=False)
            (output_dir / f"{name}.json").write_text(table.to_json(orient="records", indent=2, double_precision=15))
        summary.to_csv(output_dir / "summary.csv", index=False)
        metadata = {
            "structure_file": str(structure_path),
            "input_sha256": hashlib.sha256(structure_path.read_bytes()).hexdigest(),
            "python": sys.executable,
            "parameters": parameters,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return {
        "grid": grid, "voronoi": voronoi, "voronoi_raw": voronoi,
        "combined": combined, "merged": merged, "nonequivalent": nonequivalent,
        "clusters": clusters, "summary": summary, "settings": settings.to_dict(),
        **geometry_results,
    }
