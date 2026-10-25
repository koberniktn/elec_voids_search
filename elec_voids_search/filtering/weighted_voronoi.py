"""Дескрипторы Вороного, проверка первой сферы и сравнение с занятыми сайтами."""

from types import SimpleNamespace
import math

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from pymatgen.core import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ..descriptors.framework_voronoi import framework_voronoi_neighbors, FrameworkVoronoiError
from ..geometry.ionic_geometry import validate_oxidation_states
from ..symmetry.occupied_sites import get_nonequivalent_occupied_sites
from ..descriptors.coordination_shell import (
    EnvironmentMatchConfig,
    describe_framework_shell,
    compare_shell_geometry,
)
from .voronoi_coordination_config import VoronoiCoordinationConfig
from ..descriptors.voronoi_descriptors import (
    calc_cn_raw,
    calc_cn_significant,
    calc_cn_effective,
    calc_element_fraction,
    calc_distance_mean,
    calc_distance_std,
    calc_voronoi_volume,
    calc_voronoi_sphericity,
)


def _weighted_channel(neighbors):
    """Собираем взвешенные расстояния по элементам и углы по парам элементов."""

    if not neighbors:
        return None

    # Вес соседа — нормированный телесный угол его грани.
    weights = np.asarray([n['solid_angle'] for n in neighbors], dtype=float)
    weights = weights / weights.sum()
    directions = np.asarray([n['normal'] for n in neighbors])

    # Для каждой пары элементов храним отдельный угловой профиль.
    angle_profiles = {}

    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            pair = tuple(sorted((neighbors[i]['element'], neighbors[j]['element'])))
            angles, angle_weights = angle_profiles.setdefault(pair, ([], []))
            angles.append(
                float(np.degrees(np.arccos(np.clip(directions[i] @ directions[j], -1.0, 1.0)))),
            )
            angle_weights.append(weights[i] * weights[j])

    # Расстояния группируем по химическим элементам.
    profiles = {}

    for element in sorted({n['element'] for n in neighbors}):
        selected = [n for n in neighbors if n['element'] == element]
        profiles[element] = (
            np.asarray([n['distance'] for n in selected]),
            np.asarray([n['solid_angle'] for n in selected]),
        )

    return {
        'effective_cn': calc_cn_effective(neighbors),
        'fractions': {element: calc_element_fraction(neighbors, element) for element in profiles},
        'distance_profiles': profiles,
        'angle_profiles': {
            pair: (np.asarray(angles), np.asarray(angle_weights))
            for (pair, (angles, angle_weights)) in angle_profiles.items()
        },
    }


def describe_weighted_voronoi(
    structure,
    frac_point,
    oxidation_states,
    probe_element='Li',
    probe_oxidation_state=1,
    config=None,
):
    """Рассчитываем ячейку каркаса, дескрипторы двух каналов и первую сферу."""
    settings = config or VoronoiCoordinationConfig()

    # Строим полную ячейку каркаса без подвижных ионов.
    neighbors, cell = framework_voronoi_neighbors(structure, frac_point, probe_element, settings)

    # Основной канал — соседи с противоположным зарядом.
    primary = [n for n in neighbors if oxidation_states[n['element']] * probe_oxidation_state < 0]

    if not primary and settings.primary_environment_weight > 0:
        raise FrameworkVoronoiError('No opposite-charge framework neighbors in the cell')
    volume = calc_voronoi_volume(neighbors)

    if not np.isfinite(volume) or volume <= 0:
        raise FrameworkVoronoiError('Nonpositive or nonfinite cell volume')

    # Объём, форма и координационные показатели сохраняются раздельно.
    scalars = dict(
        cn_raw=calc_cn_raw(neighbors),
        cn_significant=calc_cn_significant(neighbors, settings.significant_face_fraction),
        cn_effective=calc_cn_effective(neighbors),
        distance_mean_A=calc_distance_mean(neighbors),
        distance_std_A=calc_distance_std(neighbors),
        voronoi_volume_A3=volume,
        voronoi_length_A=volume ** (1 / 3),
        voronoi_area_A2=float(sum((n['area'] for n in neighbors))),
        voronoi_sphericity=calc_voronoi_sphericity(neighbors),
        primary_cn_effective=calc_cn_effective(primary) if primary else np.nan,
        **cell,
    )

    try:
        # Первая сфера строится одинаково для кандидатов и занятых эталонов.
        shell = describe_framework_shell(
            structure,
            frac_point,
            oxidation_states,
            probe_element,
            probe_oxidation_state,
            EnvironmentMatchConfig(
                neighbor_cutoff_A=cell['cutoff_A'],
                shell_distance_ratio=settings.first_shell_distance_ratio,
                numeric_tolerance=settings.numeric_tolerance,
            ),
        )
    except ValueError as error:
        raise FrameworkVoronoiError(f'First coordination shell: {error}') from error
    scalars.update(
        first_shell_coordination=shell['coordination'],
        first_shell_composition=str(shell['composition']),
    )

    return {
        'full': _weighted_channel(neighbors),
        'primary': _weighted_channel(primary),
        'first_shell': shell,
        'scalars': scalars,
    }


def _channel_difference(left, right):
    """Сравниваем состав, эффективное КЧ, расстояния и поэлементные угловые профили."""
    elements = set(left['fractions']) | set(right['fractions'])
    chemistry = 0.5 * sum(
        (abs(left['fractions'].get(element, 0.0) - right['fractions'].get(element, 0.0)) for element in elements),
    )

    # Расстояния сравниваем только внутри одного элемента.
    terms, masses = ([], [])

    for element in sorted(set(left['distance_profiles']) & set(right['distance_profiles'])):
        a, wa = left['distance_profiles'][element]
        b, wb = right['distance_profiles'][element]
        mass = 0.5 * (left['fractions'][element] + right['fractions'][element])
        terms.append(mass * wasserstein_distance(a, b, u_weights=wa, v_weights=wb))
        masses.append(mass)
    distance = float(sum(terms) / sum(masses)) if masses else np.inf

    # Углы сравниваем внутри каждой пары элементов, затем взвешиваем различия.
    left_angles, right_angles = (left['angle_profiles'], right['angle_profiles'])

    if left_angles and right_angles:
        left_total = sum((weights.sum() for _, weights in left_angles.values()))
        right_total = sum((weights.sum() for _, weights in right_angles.values()))
        angular = 0.0

        for pair in sorted(set(left_angles) | set(right_angles)):
            left_mass = left_angles[pair][1].sum() / left_total if pair in left_angles else 0.0
            right_mass = right_angles[pair][1].sum() / right_total if pair in right_angles else 0.0

            if pair in left_angles and pair in right_angles:
                a, wa = left_angles[pair]
                b, wb = right_angles[pair]
                difference = wasserstein_distance(a, b, u_weights=wa, v_weights=wb)
            else:
                difference = 180.0
            angular += 0.5 * (left_mass + right_mass) * difference
        angular = float(angular)
    else:
        angular = 0.0 if not left_angles and (not right_angles) else 180.0

    return {
        'chemistry': float(chemistry),
        'effective_cn': abs(left['effective_cn'] - right['effective_cn']),
        'distance_A': distance,
        'angular_deg': angular,
    }


def compare_weighted_voronoi(candidate, reference, config=None):
    """Проверяем первую сферу и ограничения Вороного для одной пары окружений."""
    settings = config or VoronoiCoordinationConfig()

    # Первая сфера должна пройти все обязательные проверки.
    shell = compare_shell_geometry(
        candidate['first_shell'],
        reference['first_shell'],
        EnvironmentMatchConfig(
            radial_rms_tolerance_A=settings.first_shell_radial_tolerance_A,
            angle_rms_tolerance_deg=settings.first_shell_angular_tolerance_deg,
            numeric_tolerance=settings.numeric_tolerance,
        ),
    )
    shell_score = shell['normalized_error'] if shell['normalized_error'] is not None else np.inf

    if (
        (settings.primary_environment_weight > 0)
        and (candidate['primary'] is None or reference['primary'] is None)
    ):
        raise FrameworkVoronoiError('Opposite-charge environment is unavailable')

    # Сравниваем весь каркас и основной анионный канал.
    full = _channel_difference(candidate['full'], reference['full'])
    alpha = settings.primary_environment_weight
    primary = _channel_difference(candidate['primary'], reference['primary']) if alpha > 0 else full


    def mix(key):
        # Не умножаем бесконечное различие на нулевой вес.
        parts = [(alpha, primary[key]), (1 - alpha, full[key])]

        return sum((weight * value for weight, value in parts if weight > 0))

    # Нормируем каждый блок на свой открытый допуск.
    coordination = max(
        mix('chemistry') / settings.chemical_fraction_tolerance,
        mix('effective_cn') / settings.effective_cn_tolerance,
    )
    distance = mix('distance_A') / settings.distance_tolerance_A
    size = (
        abs(candidate['scalars']['voronoi_length_A'] - reference['scalars']['voronoi_length_A'])
        / settings.size_length_tolerance_A
    )
    shape = max(
        mix('angular_deg') / settings.angular_tolerance_deg,
        (
            abs(candidate['scalars']['voronoi_sphericity'] - reference['scalars']['voronoi_sphericity'])
            / settings.sphericity_tolerance
        ),
    )

    # Средняя оценка дополняет, но не заменяет обязательные ограничения.
    blocks = np.asarray([coordination, distance, size, shape])
    weights = np.asarray(settings.block_weights, dtype=float)
    active = weights > 0
    score = float(np.dot(blocks[active], weights[active]) / weights[active].sum())

    # Основные различия нельзя скрыть совпадением объёма или формы.
    primary_score = max(
        primary['chemistry'] / settings.chemical_fraction_tolerance,
        primary['effective_cn'] / settings.effective_cn_tolerance,
        primary['distance_A'] / settings.distance_tolerance_A,
        primary['angular_deg'] / settings.angular_tolerance_deg,
    ) if alpha > 0 else 0.0
    maximum_block = float(blocks[active].max())
    passed = (
        shell['similar']
        and (score <= settings.combined_score_limit + settings.numeric_tolerance)
        and (maximum_block <= settings.maximum_block_score + settings.numeric_tolerance)
        and (primary_score <= settings.maximum_primary_score + settings.numeric_tolerance)
    )

    # Сохраняем все причины отказа выбранной пары.
    reasons = []

    if not shell['similar']:
        reasons.append('first_shell_' + shell['reason'])

    if score > settings.combined_score_limit + settings.numeric_tolerance:
        reasons.append('combined_score')

    if maximum_block > settings.maximum_block_score + settings.numeric_tolerance:
        reasons.append('block_guard')

    if primary_score > settings.maximum_primary_score + settings.numeric_tolerance:
        reasons.append('primary_guard')

    return dict(
        similar=bool(passed),
        coordination_score=float(coordination),
        distance_score=float(distance),
        size_score=float(size),
        shape_score=float(shape),
        combined_score=score,
        primary_score=float(primary_score),
        maximum_block_score=maximum_block,
        primary_distance_difference_A=primary['distance_A'],
        full_distance_difference_A=full['distance_A'],
        primary_angular_difference_deg=primary['angular_deg'],
        first_shell_score=float(shell_score),
        first_shell_reason=shell['reason'],
        first_shell_radial_rms_A=shell['radial_rms_A'],
        first_shell_angle_rms_deg=shell['angle_rms_deg'],
        first_shell_reference_composition=str(reference['first_shell']['composition']),
        reason='similar_environment' if passed else ', '.join(reasons),
    )


class WeightedVoronoiMatcher:
    """Кэшируем дескрипторы и сравниваем кандидатов с занятыми симметрийными типами."""


    def __init__(
        self,
        structure,
        oxidation_states,
        probe_element='Li',
        probe_oxidation_state=1,
        config=None,
    ):
        self.config = config or VoronoiCoordinationConfig()

        if not isinstance(self.config, VoronoiCoordinationConfig):
            raise TypeError('config must be VoronoiCoordinationConfig')
        Element(probe_element)

        if not math.isfinite(probe_oxidation_state) or probe_oxidation_state == 0:
            raise ValueError('probe_oxidation_state must be finite and nonzero')
        self.structure = structure.copy()
        self.oxidation_states = dict(oxidation_states)
        validate_oxidation_states(
            self.structure,
            SimpleNamespace(geometry_oxidation_states=self.oxidation_states),
        )

        if not all(
            ((
                isinstance(value, (int, float))
                and (not isinstance(value, bool))
                and math.isfinite(value)
            ) for value in self.oxidation_states.values()),
        ):
            raise ValueError('Oxidation states must be finite numbers')
        self.probe_element = probe_element
        self.probe_oxidation_state = probe_oxidation_state

        # Дескриптор каждой точки рассчитываем один раз.
        self.cache = {}

        # Занятые сайты объединяем по симметрии.
        analyzer = SpacegroupAnalyzer(
            self.structure,
            symprec=self.config.symmetry_symprec_A,
            angle_tolerance=self.config.symmetry_angle_tolerance_deg,
        )
        self.occupied_groups = get_nonequivalent_occupied_sites(self.structure, analyzer, probe_element)

        if self.occupied_groups.empty:
            raise ValueError('No occupied probe-element reference sites')
        self.references, self.reference_errors = ({}, {})

        # Недоступные эталоны сохраняем с причиной ошибки.
        records = []

        for _, group in self.occupied_groups.iterrows():
            index = int(group.atom_indices[0])
            row = dict(
                site_index=index,
                atom_indices=group.atom_indices,
                multiplicity_in_cell=int(group.multiplicity_in_cell),
            )
            point = self.structure[index].frac_coords
            row.update({f'frac_{axis}': float(value) for axis, value in zip('xyz', point)})

            try:
                descriptor = self.describe(point)
                self.references[index] = descriptor
                row.update(
                    descriptor_status='complete',
                    descriptor_issue='',
                    **descriptor['scalars'],
                )
            except FrameworkVoronoiError as error:
                if self.config.reference_error_policy == 'raise':
                    raise
                self.reference_errors[index] = str(error)
                row.update(descriptor_status='error', descriptor_issue=str(error))
            records.append(row)
        self.reference_table = pd.DataFrame(records)


    def describe(self, point):
        key = tuple(np.asarray(point, dtype=float) % 1.0)

        if key not in self.cache:
            self.cache[key] = describe_weighted_voronoi(
                self.structure,
                key,
                self.oxidation_states,
                self.probe_element,
                self.probe_oxidation_state,
                self.config,
            )

        return self.cache[key]


    def screen(self, candidates, comparison_config=None):
        settings = comparison_config or self.config

        # Параметры построения нельзя менять у уже кэшированных дескрипторов.
        construction = [
            'initial_cutoff_A',
            'maximum_cutoff_A',
            'cutoff_growth',
            'minimum_face_area_A2',
            'solid_angle_sum_tolerance',
            'numeric_tolerance',
            'significant_face_fraction',
            'symmetry_symprec_A',
            'symmetry_angle_tolerance_deg',
            'reference_error_policy',
            'first_shell_distance_ratio',
        ]

        if any((getattr(settings, key) != getattr(self.config, key) for key in construction)):
            raise ValueError('Changed construction settings require a new matcher')

        if settings.primary_environment_weight > 0 and self.config.primary_environment_weight == 0:
            if any((d['primary'] is None for d in self.references.values())):
                raise ValueError(
                    'Primary neighbors unavailable; create a matcher with primary weight > 0',
                )
        required = {'candidate_id', 'frac_x', 'frac_y', 'frac_z'}

        if required - set(candidates.columns):
            raise ValueError(f'Missing candidate columns: {sorted(required - set(candidates.columns))}')

        # Одинаковая схема результата для всех кандидатов.
        extra = [
            'coordination_status',
            'coordination_reason',
            'coordination_reference_site_index',
            'coordination_matching_site_indices',
            'coordination_reference_errors',
            'coordination_score',
            'coordination_distance_score',
            'coordination_size_score',
            'coordination_shape_score',
            'coordination_combined_score',
            'coordination_primary_score',
            'coordination_primary_distance_difference_A',
            'coordination_primary_angular_difference_deg',
            'coordination_voronoi_volume_A3',
            'coordination_voronoi_sphericity',
            'coordination_effective_cn',
            'coordination_primary_effective_cn',
            'coordination_cutoff_A',
        ]
        extra += [
            'coordination_first_shell_coordination',
            'coordination_first_shell_composition',
            'coordination_first_shell_reference_composition',
            'coordination_first_shell_score',
            'coordination_first_shell_reason',
            'coordination_first_shell_radial_rms_A',
            'coordination_first_shell_angle_rms_deg',
        ]

        # Достаточно подтверждённого совпадения хотя бы с одним занятым типом.
        rows = []

        for _, candidate in candidates.iterrows():
            row = candidate.to_dict()
            row.update({column: None for column in extra})
            row['coordination_reference_errors'] = dict(self.reference_errors)

            try:
                descriptor = self.describe(candidate[['frac_x', 'frac_y', 'frac_z']].to_numpy(dtype=float))
                pairs = [
                    (index, compare_weighted_voronoi(descriptor, reference, settings))
                    for (index, reference) in self.references.items()
                ]
                matches = [pair for pair in pairs if pair[1]['similar']]
                status = 'passed' if matches else 'indeterminate' if self.reference_errors else 'rejected'
                row.update(
                    coordination_status=status,
                    coordination_matching_site_indices=[pair[0] for pair in matches],
                    coordination_reason='incomplete_references' if status == 'indeterminate' else 'no_matching_reference',
                    coordination_voronoi_volume_A3=descriptor['scalars']['voronoi_volume_A3'],
                    coordination_voronoi_sphericity=descriptor['scalars']['voronoi_sphericity'],
                    coordination_effective_cn=descriptor['scalars']['cn_effective'],
                    coordination_primary_effective_cn=descriptor['scalars']['primary_cn_effective'],
                    coordination_first_shell_coordination=descriptor['first_shell']['coordination'],
                    coordination_first_shell_composition=str(descriptor['first_shell']['composition']),
                    coordination_cutoff_A=descriptor['scalars']['cutoff_A'],
                )

                if pairs:

                    def rank(pair):
                        values = pair[1]

                        return max(
                            values['combined_score'] / settings.combined_score_limit,
                            values['maximum_block_score'] / settings.maximum_block_score,
                            values['primary_score'] / settings.maximum_primary_score,
                            values['first_shell_score'],
                        )
                    index, best = min(matches or pairs, key=rank)
                    row.update(
                        coordination_reference_site_index=index,
                        coordination_reason='incomplete_references' if status == 'indeterminate' else best['reason'],
                        coordination_score=best['coordination_score'],
                        coordination_distance_score=best['distance_score'],
                        coordination_size_score=best['size_score'],
                        coordination_shape_score=best['shape_score'],
                        coordination_combined_score=best['combined_score'],
                        coordination_primary_score=best['primary_score'],
                        coordination_primary_distance_difference_A=best['primary_distance_difference_A'],
                        coordination_primary_angular_difference_deg=best['primary_angular_difference_deg'],
                    )
                    row.update(
                        {
                            f'coordination_{key}': best[key]
                            for key in [
                                'first_shell_reference_composition',
                                'first_shell_score',
                                'first_shell_reason',
                                'first_shell_radial_rms_A',
                                'first_shell_angle_rms_deg',
                            ]
                        },
                    )
            except FrameworkVoronoiError as error:
                row.update(coordination_status='indeterminate', coordination_reason=str(error))
            rows.append(row)

        return pd.DataFrame(
            rows,
            columns=list(candidates.columns) + [key for key in extra if key not in candidates.columns],
        )


def screen_voronoi_coordination(
    structure,
    candidates,
    oxidation_states,
    probe_element='Li',
    probe_oxidation_state=1,
    config=None,
):
    """Разделяем кандидатов на прошедшие, отклонённые и неопределённые."""
    matcher = WeightedVoronoiMatcher(
        structure,
        oxidation_states,
        probe_element,
        probe_oxidation_state,
        config,
    )
    evaluated = matcher.screen(candidates)

    return dict(
        coordination_evaluated=evaluated,
        coordination_passed=evaluated[evaluated.coordination_status.eq('passed')].copy(),
        coordination_rejected=evaluated[evaluated.coordination_status.eq('rejected')].copy(),
        coordination_indeterminate=evaluated[evaluated.coordination_status.eq('indeterminate')].copy(),
        occupied_reference_descriptors=matcher.reference_table,
        settings=matcher.config.to_dict(),
    )
