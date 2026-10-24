"""Geometry screening after symmetry grouping, with full candidate lineage."""
import numpy as np
import pandas as pd

from .ionic_geometry import prepare_ionic_geometry, ProbeRadiusCalculator, occupied_probe_controls
from .radius_geometry import make_point_geometry


def screen_geometry(structure, nonequivalent, settings):
    """Determine each host radius and each pore's environment-dependent probe.

    Only preselected symmetry representatives are evaluated. Raw merged-away
    candidates are not reintroduced. Coordinates are never changed here.
    """
    if not len(structure) or not structure.is_ordered:
        raise ValueError("Geometry requires a nonempty ordered structure")
    charged, sites = prepare_ionic_geometry(structure, settings)
    radii = sites.radius_A.to_numpy(dtype=float)
    if not np.isfinite(radii).all():
        raise ValueError("Nonfinite host radius: supply a radius override")
    point_geometry = make_point_geometry(structure, dict(enumerate(radii)))
    probe_radius = ProbeRadiusCalculator(charged, settings)
    tolerance = settings.geometry_overlap_tolerance_A + settings.geometry_numeric_tolerance_A

    def evaluate(point):
        free_radius, nearest_distance, limiting_atom = point_geometry(point)
        probe = probe_radius(point)
        if not np.isfinite(probe['probe_radius_A']):
            raise ValueError('Nonfinite probe radius: supply geometry_probe_radius_override_A')
        required = settings.geometry_minimum_fit_fraction * probe['probe_radius_A']
        margin = free_radius - required
        strict_margin = free_radius - probe['probe_radius_A']
        status = 'rejected' if margin < -tolerance else 'passed'
        strict_status = 'rejected' if strict_margin < -tolerance else 'passed'
        return dict(**probe, free_radius_A=free_radius, minimum_free_radius_A=required,
                    free_radius_complete=bool(np.isfinite(radii).all()),
                    nearest_atom_center_A=nearest_distance, geometry_margin_A=margin,
                    full_radius_margin_A=strict_margin, geometry_status=status,
                    full_radius_status=strict_status,
                    strict_zero_overlap_fit=bool(strict_margin >= -settings.geometry_numeric_tolerance_A),
                    geometry_allowed_overlap_A=settings.geometry_overlap_tolerance_A,
                    geometry_limiting_site_index=int(limiting_atom) if limiting_atom >= 0 else None,
                    geometry_limiting_element=sites.iloc[limiting_atom].element if limiting_atom >= 0 else None,
                    geometry_unresolved_sites=int((~np.isfinite(radii)).sum()),
                    geometry_reason='known_obstacle_overlap' if status == 'rejected' else 'ion_fits_radius_model')

    audit, rows = [], []
    for _, representative in nonequivalent.iterrows():
        point = representative[['frac_x', 'frac_y', 'frac_z']].to_numpy(dtype=float)
        data = evaluate(point)
        row = representative.to_dict()
        row.update(data)
        rows.append(row)
        audit.append(dict(symmetry_group_id=int(representative.symmetry_group_id),
                          candidate_id=int(representative.candidate_id),
                          source_method=representative.source_method,
                          frac_x=point[0], frac_y=point[1], frac_z=point[2], **data))
    decision_columns = ['probe_element', 'probe_oxidation_state', 'probe_coordination',
                        'probe_radius_A', 'probe_radius_source', 'probe_radius_issue', 'probe_radius_warning',
                        'probe_radius_reference_coordinations', 'probe_radius_reference_values_A',
                        'probe_radius_requested_coordination', 'probe_radius_approximate', 'probe_radius_convention',
                        'free_radius_A', 'minimum_free_radius_A', 'geometry_status',
                        'full_radius_status', 'geometry_margin_A', 'full_radius_margin_A',
                        'strict_zero_overlap_fit', 'geometry_allowed_overlap_A',
                        'free_radius_complete', 'geometry_reason', 'geometry_unresolved_sites',
                        'geometry_limiting_site_index', 'geometry_limiting_element']
    columns = list(dict.fromkeys(list(nonequivalent.columns) + decision_columns))
    evaluated = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    return dict(geometry_evaluated=evaluated,
                geometry=evaluated[evaluated.geometry_status == 'passed'].copy(),
                geometry_indeterminate=evaluated[evaluated.geometry_status == 'indeterminate'].copy(),
                geometry_rejected=evaluated[evaluated.geometry_status == 'rejected'].copy(),
                geometry_retained=evaluated[evaluated.geometry_status != 'rejected'].copy(),
                geometry_candidate_audit=pd.DataFrame(audit, columns=list(dict.fromkeys(
                    ['symmetry_group_id', 'candidate_id', 'source_method',
                     'frac_x', 'frac_y', 'frac_z'] + decision_columns))),
                site_radii=sites,
                occupied_probe_controls=occupied_probe_controls(charged, sites, settings))
