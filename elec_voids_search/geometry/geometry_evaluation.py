"""Общий радиусный оценщик для кандидатов и освобождённых занятых сайтов."""

import numpy as np

from .ionic_geometry import prepare_ionic_geometry, ProbeRadiusCalculator
from .radius_geometry import make_point_geometry


def make_ionic_geometry_evaluator(structure, settings):
    """Определяем радиусы каркаса и возвращаем оценщик точек."""

    if not len(structure) or not structure.is_ordered:
        raise ValueError('Geometry requires a nonempty ordered structure')

    # Радиусы соседей определяем отдельно для каждого сайта.
    charged, sites = prepare_ionic_geometry(structure, settings)
    radii = sites.radius_A.to_numpy(dtype=float)

    if not np.isfinite(radii).all():
        raise ValueError('Nonfinite host radius: supply a radius override')
    point_geometry = make_point_geometry(structure, dict(enumerate(radii)))
    probe_radius = ProbeRadiusCalculator(charged, settings)


    def evaluate(point):
        free_radius, nearest_distance, limiting_atom = point_geometry(point)

        # Радиус пробника зависит от окружения этой полости.
        probe = probe_radius(point)

        if not np.isfinite(probe['probe_radius_A']):
            raise ValueError('Nonfinite probe radius: supply geometry_probe_radius_override_A')

        # Процентный допуск считаем от радиуса пробного иона.
        allowed_overlap = settings.geometry_allowed_overlap_A(probe['probe_radius_A'])
        tolerance = allowed_overlap + settings.geometry_numeric_tolerance_A
        required = settings.geometry_minimum_fit_fraction * probe['probe_radius_A']
        margin = free_radius - required

        # Строгий зазор сохраняем отдельно от решения с допуском.
        strict_margin = free_radius - probe['probe_radius_A']
        status = 'rejected' if margin < -tolerance else 'passed'
        strict_status = 'rejected' if strict_margin < -tolerance else 'passed'

        return dict(
            **probe,
            free_radius_A=free_radius,
            minimum_free_radius_A=required,
            free_radius_complete=bool(np.isfinite(radii).all()),
            nearest_atom_center_A=nearest_distance,
            geometry_margin_A=margin,
            full_radius_margin_A=strict_margin,
            geometry_status=status,
            full_radius_status=strict_status,
            strict_zero_overlap_fit=bool(strict_margin >= -settings.geometry_numeric_tolerance_A),
            geometry_allowed_overlap_A=allowed_overlap,
            geometry_allowed_overlap_percent=settings.geometry_overlap_tolerance_percent,
            geometry_limiting_site_index=int(limiting_atom) if limiting_atom >= 0 else None,
            geometry_limiting_element=sites.iloc[limiting_atom].element if limiting_atom >= 0 else None,
            geometry_unresolved_sites=int((~np.isfinite(radii)).sum()),
            geometry_reason='known_obstacle_overlap' if status == 'rejected' else 'ion_fits_radius_model',
        )

    return (charged, sites, evaluate)
