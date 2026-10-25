"""Геометрический отсев после симметрии без изменения координат."""

import numpy as np
import pandas as pd

from .ionic_geometry import occupied_probe_controls
from .geometry_evaluation import make_ionic_geometry_evaluator


def screen_geometry(structure, nonequivalent, settings):
    """Проверяем размер пор"""

    # Находим радиусы каркаса.
    charged, sites, evaluate = make_ionic_geometry_evaluator(structure, settings)

    # Не загружаем экспериментальную калибровку в процентном режиме.
    calibration = None

    if settings.geometry_reference_mode == 'occupied_sites':
        from .geometry_references import OccupiedGeometryCalibration
        calibration = OccupiedGeometryCalibration(structure, settings)

    # Оцениваем только выбранных симметрийных представителей.
    audit, rows = ([], [])

    for _, representative in nonequivalent.iterrows():
        point = representative[['frac_x', 'frac_y', 'frac_z']].to_numpy(dtype=float)
        data = evaluate(point)

        if calibration is not None:
            data.update(calibration.evaluate(point, data))
        row = representative.to_dict()
        row.update(data)
        rows.append(row)
        audit.append(
            dict(
                symmetry_group_id=int(representative.symmetry_group_id),
                candidate_id=int(representative.candidate_id),
                source_method=representative.source_method,
                frac_x=point[0],
                frac_y=point[1],
                frac_z=point[2],
                **data,
            ),
        )

    # Единая схема полей для непустых и пустых результатов.
    decision_columns = [
        'probe_element',
        'probe_oxidation_state',
        'probe_coordination',
        'probe_radius_A',
        'probe_radius_source',
        'probe_radius_issue',
        'probe_radius_warning',
        'probe_radius_reference_coordinations',
        'probe_radius_reference_values_A',
        'probe_radius_requested_coordination',
        'probe_radius_approximate',
        'probe_radius_convention',
        'free_radius_A',
        'minimum_free_radius_A',
        'geometry_status',
        'full_radius_status',
        'geometry_margin_A',
        'full_radius_margin_A',
        'strict_zero_overlap_fit',
        'geometry_allowed_overlap_A',
        'geometry_allowed_overlap_percent',
        'free_radius_complete',
        'geometry_reason',
        'geometry_unresolved_sites',
        'geometry_limiting_site_index',
        'geometry_limiting_element',
    ]

    if calibration is not None:
        decision_columns += calibration.decision_columns
    columns = list(dict.fromkeys(list(nonequivalent.columns) + decision_columns))
    evaluated = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)

    return dict(
        geometry_evaluated=evaluated,
        geometry=evaluated[evaluated.geometry_status == 'passed'].copy(),
        geometry_indeterminate=evaluated[evaluated.geometry_status == 'indeterminate'].copy(),
        geometry_rejected=evaluated[evaluated.geometry_status == 'rejected'].copy(),
        geometry_retained=evaluated[evaluated.geometry_status != 'rejected'].copy(),
        geometry_candidate_audit=pd.DataFrame(
            audit,
            columns=list(
                dict.fromkeys(
                    [
                        'symmetry_group_id',
                        'candidate_id',
                        'source_method',
                        'frac_x',
                        'frac_y',
                        'frac_z',
                    ] + decision_columns,
                ),
            ),
        ),
        occupied_geometry_references=calibration.references.copy() if calibration is not None else pd.DataFrame(),
        site_radii=sites,
        occupied_probe_controls=occupied_probe_controls(charged, sites, settings),
    )
