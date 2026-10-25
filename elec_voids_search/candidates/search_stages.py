"""Общие стадии поиска кандидатов и сбор результатов геометрии."""

import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .combination import combine_candidates
from .grid_search import grid_candidates
from .merging import merge_candidates
from ..symmetry.periodic_groups import periodic_groups
from .voronoi_search import voronoi_candidates


def find_candidate_sites(structure, structure_path, settings):
    """Сырые кандидаты → общая кластеризация → симметрийные представители."""

    # 1. Независимые сырые кандидаты сетки и Вороного.
    grid, grid_parameters = grid_candidates(structure_path, config=settings, structure=structure)
    voronoi, voronoi_parameters, voronoi_details = voronoi_candidates(structure, config=settings, return_details=True)

    # 2. Объединяем полные списки двух методов.
    combined = combine_candidates(grid, voronoi)

    # 3. Общая кластеризация близких точек.
    combined, merged, clusters = merge_candidates(
        structure,
        combined,
        distance=settings.merge_distance_A,
        representative=settings.merge_representative,
        grouping=settings.merge_grouping,
    )

    # 4. Симметрийный отсев до проверки геометрии.
    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=settings.symmetry_symprec_A,
        angle_tolerance=settings.symmetry_angle_tolerance_deg,
    )
    positions = merged[['frac_x', 'frac_y', 'frac_z']].to_numpy()
    groups = periodic_groups(
        structure,
        positions,
        settings.symmetry_tolerance_A,
        operations=analyzer.get_symmetry_operations(cartesian=False),
    )
    merged['symmetry_group_id'] = pd.Series(index=merged.index, dtype='int64')

    for group_id, indices in enumerate(groups):
        merged.loc[merged.index[indices], 'symmetry_group_id'] = group_id
    merged['symmetry_group_id'] = merged.symmetry_group_id.astype(int)
    group_by_cluster = merged.set_index('merge_cluster_id').symmetry_group_id
    combined['symmetry_group_id'] = combined.merge_cluster_id.map(group_by_cluster).astype(int)
    representative_rows = [merged.index[indices[0]] for indices in groups]
    nonequivalent = merged.loc[representative_rows].copy()
    selected_ids = set(nonequivalent.candidate_id)

    for table in [combined, merged, nonequivalent]:
        table['is_symmetry_representative'] = table.candidate_id.isin(selected_ids)

    return dict(
        grid=grid, voronoi=voronoi, combined=combined, merged=merged,
        nonequivalent=nonequivalent, clusters=clusters,
        grid_parameters=grid_parameters, voronoi_parameters=voronoi_parameters,
        voronoi_details=voronoi_details, space_group=analyzer.get_space_group_symbol(),
    )


def assemble_search_result(structure, structure_path, settings, sites, geometry_results):
    """Собираем ранние стадии и геометрию"""

    grid, voronoi = sites['grid'], sites['voronoi']
    combined, merged = sites['combined'], sites['merged']
    nonequivalent, clusters = sites['nonequivalent'], sites['clusters']
    grid_parameters = sites['grid_parameters']
    voronoi_parameters = sites['voronoi_parameters']

    # Итоговые количества и фактически использованные настройки.
    summary = pd.DataFrame(
        [
            {
                'structure_file': structure_path.name,
                'formula': structure.composition.reduced_formula,
                'space_group': sites['space_group'],
                'n_grid_candidates': len(grid),
                'n_voronoi_raw_vertices': len(voronoi),
                'n_voronoi_candidates': len(voronoi),
                'n_combined_candidates': len(combined),
                'n_merged_candidates': len(merged),
                'n_symmetry_nonequivalent_candidates': len(nonequivalent),
                'n_geometry_passed': len(geometry_results['geometry']),
                'n_geometry_indeterminate': len(geometry_results['geometry_indeterminate']),
                'n_geometry_rejected': len(geometry_results['geometry_rejected']),
                'n_geometry_retained': len(geometry_results['geometry_retained']),
                'n_full_radius_passed': int((geometry_results['geometry_evaluated'].full_radius_status == 'passed').sum()),
                'probe_element': settings.probe_element,
                'probe_oxidation_state': settings.probe_oxidation_state,
                'geometry_minimum_fit_fraction': settings.geometry_minimum_fit_fraction,
                'geometry_reference_mode': settings.geometry_reference_mode,
                'n_geometry_reference_types': len(geometry_results['occupied_geometry_references']),
                'n_geometry_rescued': int(
                    (
                        geometry_results['geometry_evaluated'].geometry_status.eq('passed')
                        & geometry_results['geometry_evaluated'].get(
                            'geometry_uncalibrated_status',
                            geometry_results['geometry_evaluated'].geometry_status,
                        ).eq('rejected')
                    ).sum(),
                ),
                'grid_resolution_A': settings.grid_resolution_A,
                'merge_distance_A': settings.merge_distance_A,
                'merge_representative': settings.merge_representative,
                'merge_grouping': settings.merge_grouping,
                'symprec_A': settings.symmetry_symprec_A,
                'symmetry_tolerance_A': settings.symmetry_tolerance_A,
            },
        ],
    )
    parameters = {
        'settings': settings.to_dict(),
        'grid': grid_parameters,
        'voronoi': voronoi_parameters,
        'stage_order': ['grid_and_voronoi', 'combined', 'merged', 'nonequivalent', 'geometry'],
        'combination_order': ['grid', 'voronoi'],
        'radius_filter_applied': True,
        'periodic_grouping': 'exact_periodic_distances',
        'pre_geometry_merge_grouping': settings.merge_grouping,
        'geometry_coordinates': 'fixed_preselected_symmetry_representatives',
        'symmetry_reduction_stage': 'before_geometry_only',
    }

    return {
        'grid': grid,
        'voronoi': voronoi,
        'voronoi_raw': voronoi,
        'combined': combined,
        'merged': merged,
        'nonequivalent': nonequivalent,
        'clusters': clusters,
        'summary': summary,
        'settings': settings.to_dict(),
        'parameters': parameters,
        'raw_voronoi_vertices': sites['voronoi_details']['raw_voronoi_df'],
        **geometry_results,
    }
