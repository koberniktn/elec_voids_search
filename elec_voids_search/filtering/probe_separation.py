"""Отбор по расстоянию до занятого пробного иона; координаты неизменны."""

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd
from pymatgen.core import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ..symmetry.occupied_sites import get_nonequivalent_occupied_sites, site_has_element


@dataclass(frozen=True)
class ProbeSeparationConfig:
    """Допуск расстояния до занятого пробного иона и параметры симметрии."""
    probe_element: str = 'Li'
    distance_tolerance_A: float = 0.1
    numeric_tolerance_A: float = 1e-08
    symmetry_symprec_A: float = 0.1
    symmetry_angle_tolerance_deg: float = 5.0


    def __post_init__(self):
        Element(self.probe_element)

        for name in ['distance_tolerance_A', 'numeric_tolerance_A']:
            value = getattr(self, name)

            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')

        if not math.isfinite(self.symmetry_symprec_A) or self.symmetry_symprec_A <= 0:
            raise ValueError('symmetry_symprec_A must be finite and positive')
        angle = self.symmetry_angle_tolerance_deg

        if not math.isfinite(angle) or not (angle == -1 or 0 < angle <= 180):
            raise ValueError('Invalid symmetry_angle_tolerance_deg')


    def to_dict(self):
        return asdict(self)


def occupied_probe_distances(structure, probe_element='Li'):
    """Находим ближайший другой пробный ион для каждого занятого сайта.

    На диагонали — кратчайший ненулевой перенос решётки, поэтому учитываются
    периодические соседи даже при одном ионе в ячейке. Усредняем по всем сайтам.
    """

    if not structure.is_ordered:
        raise ValueError('Resolve partial occupancies before separation screening')
    indices = [i for i, site in enumerate(structure) if site_has_element(site, probe_element)]

    if not indices:
        raise ValueError(f'No occupied {probe_element} sites; reference distance is undefined')
    positions = structure.frac_coords[indices]

    # На диагонали учитываем ближайшую ненулевую периодическую копию.
    images = structure.lattice.get_points_in_sphere(
        [[0.0, 0.0, 0.0]],
        [0.0, 0.0, 0.0],
        min(structure.lattice.abc) + 1e-08,
    )
    shortest_translation = min((float(item[1]) for item in images if np.any(np.asarray(item[3]) != 0)))
    distances = structure.lattice.get_all_distances(positions, positions)
    np.fill_diagonal(distances, shortest_translation)

    # Среднее считаем по всем занятым сайтам, не по представителям симметрии.
    nearest = distances.min(axis=1)
    closest = distances.argmin(axis=1)
    table = pd.DataFrame(
        {
            'site_index': indices,
            'nearest_probe_site_index': np.asarray(indices)[closest],
            'nearest_is_periodic_self_image': closest == np.arange(len(indices)),
            'nearest_probe_distance_A': nearest,
            'frac_x': positions[:, 0],
            'frac_y': positions[:, 1],
            'frac_z': positions[:, 2],
        },
    )

    return (table, float(nearest.mean()))


def screen_probe_separation(structure, geometry_candidates, config=None):
    """Проверяем расстояние до ближайшего занятого пробника без перемещения точек."""
    settings = config if config is not None else ProbeSeparationConfig()

    if not isinstance(settings, ProbeSeparationConfig):
        raise TypeError('config must be ProbeSeparationConfig')
    required = ['candidate_id', 'frac_x', 'frac_y', 'frac_z']
    missing = set(required) - set(geometry_candidates.columns)

    if missing:
        raise ValueError(f'Missing candidate columns: {sorted(missing)}')

    if (
        ('geometry_status' in geometry_candidates)
        and (not geometry_candidates.geometry_status.eq('passed').all())
    ):
        raise ValueError('Supply only candidates which passed geometry')
    occupied, mean_distance = occupied_probe_distances(structure, settings.probe_element)
    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=settings.symmetry_symprec_A,
        angle_tolerance=settings.symmetry_angle_tolerance_deg,
    )
    unique = get_nonequivalent_occupied_sites(structure, analyzer, settings.probe_element)
    unique = unique.copy()
    unique['representative_site_index'] = unique.atom_indices.map(lambda indices: indices[0])
    coords = structure.frac_coords[unique.representative_site_index.to_numpy(dtype=int)]
    unique[['frac_x', 'frac_y', 'frac_z']] = coords
    points = geometry_candidates[['frac_x', 'frac_y', 'frac_z']].to_numpy(dtype=float)

    if not np.isfinite(points).all():
        raise ValueError('Candidate coordinates must be finite')

    # Координаты кандидатов остаются неизменными.
    evaluated = geometry_candidates.copy()

    if len(points):
        distances = structure.lattice.get_all_distances(points, structure.frac_coords[occupied.site_index])
        nearest_indices = distances.argmin(axis=1)
        nearest = distances[np.arange(len(points)), nearest_indices]
        nearest_site = occupied.site_index.to_numpy()[nearest_indices]
    else:
        nearest = np.array([], dtype=float)
        nearest_site = np.array([], dtype=int)
    evaluated['separation_probe_element'] = settings.probe_element
    evaluated['nearest_occupied_probe_site_index'] = nearest_site
    evaluated['nearest_occupied_probe_distance_A'] = nearest
    evaluated['mean_probe_nearest_neighbor_A'] = mean_distance
    evaluated['probe_distance_tolerance_A'] = settings.distance_tolerance_A
    evaluated['probe_distance_margin_A'] = nearest - mean_distance
    passed = nearest + settings.distance_tolerance_A + settings.numeric_tolerance_A >= mean_distance
    evaluated['probe_distance_status'] = np.where(passed, 'passed', 'rejected')
    evaluated['screening_group'] = np.where(
        passed,
        'geometry_and_probe_distance_passed',
        'geometry_passed_probe_distance_rejected',
    )

    return {
        'probe_distance_evaluated': evaluated,
        'probe_distance_passed': evaluated.loc[passed].copy(),
        'probe_distance_rejected': evaluated.loc[~passed].copy(),
        'occupied_probe_sites': occupied,
        'occupied_probe_nonequivalent': unique,
        'summary': pd.DataFrame(
            [
                {
                    'probe_element': settings.probe_element,
                    'n_occupied_probe_sites': len(occupied),
                    'n_occupied_probe_nonequivalent': len(unique),
                    'mean_probe_nearest_neighbor_A': mean_distance,
                    'probe_distance_tolerance_A': settings.distance_tolerance_A,
                    'n_geometry_passed': len(evaluated),
                    'n_probe_distance_passed': int(passed.sum()),
                    'n_probe_distance_rejected': int((~passed).sum()),
                },
            ],
        ),
        'settings': settings.to_dict(),
    }
