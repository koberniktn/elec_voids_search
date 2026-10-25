"""Общее описание первой сферы и независимое сравнение длин связей и углов."""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EnvironmentMatchConfig:
    """Параметры первой сферы и прежнего исследовательского выравнивания."""
    neighbor_cutoff_A: float = 6.0
    shell_distance_ratio: float = 1.25
    radial_rms_tolerance_A: float = 0.15
    angle_rms_tolerance_deg: float = 12.0
    aligned_rms_tolerance_A: float = 0.25
    maximum_exact_permutations: int = 40320
    allow_reflection: bool = True
    symmetry_symprec_A: float = 0.1
    numeric_tolerance: float = 1e-08


    def __post_init__(self):
        for name in [
            'neighbor_cutoff_A',
            'radial_rms_tolerance_A',
            'angle_rms_tolerance_deg',
            'aligned_rms_tolerance_A',
            'symmetry_symprec_A',
        ]:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')

        if not math.isfinite(self.shell_distance_ratio) or self.shell_distance_ratio <= 1:
            raise ValueError('shell_distance_ratio must be > 1')

        if not math.isfinite(self.numeric_tolerance) or self.numeric_tolerance < 0:
            raise ValueError('numeric_tolerance must be finite and nonnegative')

        if (
            (not isinstance(self.maximum_exact_permutations, int))
            or isinstance(self.maximum_exact_permutations, bool)
            or (self.maximum_exact_permutations < 1)
        ):
            raise ValueError('maximum_exact_permutations must be a positive integer')

        if not isinstance(self.allow_reflection, bool):
            raise ValueError('allow_reflection must be bool')


    def to_dict(self):
        return asdict(self)


def describe_framework_shell(
    structure,
    frac_point,
    oxidation_states,
    probe_element='Li',
    probe_oxidation_state=1,
    config=None,
):
    """Находим противоположно заряженных соседей первой сферы без пробного элемента."""
    settings = config or EnvironmentMatchConfig()
    center = structure.lattice.get_cartesian_coords(frac_point)

    # Пробный элемент исключён; оставляем противоположно заряженный каркас.
    neighbors = []

    for neighbor in structure.get_sites_in_sphere(center, settings.neighbor_cutoff_A):
        symbol = neighbor.specie.symbol

        if symbol == probe_element or oxidation_states[symbol] * probe_oxidation_state >= 0:
            continue

        if neighbor.nn_distance <= settings.numeric_tolerance:
            raise ValueError('Candidate coincides with a coordinating framework atom')
        neighbors.append(
            (float(neighbor.nn_distance), symbol, np.asarray(neighbor.coords) - center),
        )

    if not neighbors:
        raise ValueError('No oppositely charged framework neighbors inside cutoff')

    # Первую сферу выделяем относительно ближайшего соседа.
    nearest = min((item[0] for item in neighbors))
    boundary = settings.shell_distance_ratio * nearest

    if boundary >= settings.neighbor_cutoff_A:
        raise ValueError('Increase neighbor_cutoff_A to cover the entire shell')
    selected = sorted(
        (item for item in neighbors if item[0] <= boundary + settings.numeric_tolerance),
        key=lambda item: (item[1], item[0], *item[2]),
    )
    labels = [item[1] for item in selected]
    vectors = np.asarray([item[2] for item in selected])
    lengths = np.linalg.norm(vectors, axis=1)
    units = vectors / lengths[:, None]

    # Углы группируем по неупорядоченным парам элементов.
    angles = {}

    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            key = tuple(sorted((labels[i], labels[j])))
            angles.setdefault(key, []).append(float(np.degrees(np.arccos(np.clip(units[i] @ units[j], -1.0, 1.0)))))

    return {
        'labels': labels,
        'vectors': vectors,
        'lengths': lengths,
        'angles': {key: np.sort(values) for key, values in angles.items()},
        'coordination': len(selected),
        'shell_radius_A': boundary,
        'composition': {symbol: labels.count(symbol) for symbol in sorted(set(labels))},
    }


def compare_shell_geometry(candidate, reference, config=None):
    """Требуем одинаковый состав сферы и отдельно проверяем длины связей и углы."""
    settings = config or EnvironmentMatchConfig()
    result = {
        'similar': False,
        'comparison_status': 'rejected',
        'reason': 'shell_composition',
        'radial_rms_A': None,
        'angle_rms_deg': None,
        'aligned_rms_A': None,
        'normalized_error': None,
    }

    # Разный состав первой сферы сразу означает несовпадение.
    if candidate['composition'] != reference['composition']:
        return result
    labels = candidate['labels']

    # Длины связей сравниваем по элементам без масштабирования.
    squared = []

    for symbol in sorted(set(labels)):
        left = np.sort(candidate['lengths'][np.asarray(labels) == symbol])
        right = np.sort(reference['lengths'][np.asarray(reference['labels']) == symbol])
        squared.extend((left - right) ** 2)
    radial = float(np.sqrt(np.mean(squared)))

    # Углы сравниваем отдельно по парам элементов.
    angle_sq = [
        delta ** 2
        for (key, values) in candidate['angles'].items()
        for delta in values - reference['angles'][key]
    ]
    angular = float(np.sqrt(np.mean(angle_sq))) if angle_sq else 0.0
    result.update(radial_rms_A=radial, angle_rms_deg=angular, reason='bond_lengths_or_angles')

    # Оба допуска обязательны: один признак не компенсирует другой.
    ratios = [radial / settings.radial_rms_tolerance_A, angular / settings.angle_rms_tolerance_deg]
    result['normalized_error'] = max(ratios)

    if max(ratios) > 1.0 + settings.numeric_tolerance:
        return result
    result.update(similar=True, comparison_status='passed', reason='similar_shell_geometry')

    return result
