"""Необязательная калибровка геометрии по занятым сайтам той же структуры."""

from dataclasses import replace

import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ..descriptors.coordination_shell import EnvironmentMatchConfig, describe_framework_shell
from .geometry_evaluation import make_ionic_geometry_evaluator
from ..symmetry.occupied_sites import get_nonequivalent_occupied_sites


def vacancy_settings(settings, removed_index):
    """Переносим индексные радиусы на те же атомы после удаления сайта."""
    overrides = {}

    for key, value in settings.geometry_host_radius_overrides_A.items():
        if key.isdecimal():
            index = int(key)

            if index == removed_index:
                continue
            key = str(index - int(index > removed_index))
        overrides[key] = value

    return replace(settings, geometry_host_radius_overrides_A=overrides)


def geometry_reference_type(structure, point, probe, settings):
    """Выбираем грубый тип по КЧ пробника и составу первой сферы; углы не фильтруем."""
    shell = describe_framework_shell(
        structure,
        point,
        settings.geometry_oxidation_states,
        settings.probe_element,
        settings.probe_oxidation_state,
        EnvironmentMatchConfig(
            neighbor_cutoff_A=settings.geometry_neighbor_cutoff_A,
            shell_distance_ratio=settings.geometry_reference_shell_distance_ratio,
            numeric_tolerance=settings.geometry_numeric_tolerance_A,
        ),
    )
    coordination = probe['probe_coordination']
    composition = tuple(sorted(shell['composition'].items()))

    if coordination is None or not np.isfinite(coordination):
        return (None, str(dict(composition)))

    return ((float(coordination), composition), str(dict(composition)))


class OccupiedGeometryCalibration:
    """Рассчитываем занятые симметрийные типы один раз и кэшируем типы кандидатов."""
    decision_columns = [
        'geometry_uncalibrated_status',
        'geometry_free_radius_ratio',
        'geometry_reference_status',
        'geometry_reference_composition',
        'geometry_reference_site_index',
        'geometry_reference_ratio',
        'geometry_threshold_ratio',
        'geometry_effective_minimum_free_radius_A',
        'geometry_effective_allowed_overlap_A',
        'geometry_effective_margin_A',
        'geometry_reference_applied',
        'geometry_reference_issue',
    ]


    def __init__(self, structure, settings):
        self.structure = structure
        self.settings = settings
        self.types = {}
        self.cache = {}

        # Один эталон на каждый занятый симметрийный тип.
        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=settings.symmetry_symprec_A,
            angle_tolerance=settings.symmetry_angle_tolerance_deg,
        )
        groups = get_nonequivalent_occupied_sites(structure, analyzer, settings.probe_element)
        rows, keys = ([], [])

        for _, group in groups.iterrows():
            index = int(group.atom_indices[0])
            point = structure[index].frac_coords.copy()
            vacancy = structure.copy()
            vacancy.remove_sites([index])

            if not len(vacancy):
                raise ValueError('Occupied-site calibration requires a nonempty vacancy host')
            local_settings = vacancy_settings(settings, index)
            _, _, evaluate = make_ionic_geometry_evaluator(vacancy, local_settings)
            data = evaluate(point)
            ratio = data['free_radius_A'] / data['probe_radius_A']
            key, composition, issue = (None, '', '')

            try:
                key, composition = geometry_reference_type(vacancy, point, data, local_settings)
            except ValueError as error:
                issue = str(error)
            eligible = key is not None and np.isfinite(ratio) and (ratio > 0)

            if not eligible and (not issue):
                issue = 'Unknown coordination or nonpositive free radius'
            row = dict(
                site_index=index,
                atom_indices=list(group.atom_indices),
                multiplicity_in_cell=int(group.multiplicity_in_cell),
                frac_x=float(point[0]),
                frac_y=float(point[1]),
                frac_z=float(point[2]),
                reference_composition=composition,
                reference_ratio=float(ratio),
                reference_eligible=bool(eligible),
                reference_issue=issue,
                **data,
            )

            if eligible:
                self.types.setdefault(key, []).append(row)
            rows.append(row)
            keys.append(key if eligible else None)

        # Эта таблица показывает калибровку, а не независимую валидацию.
        self.references = pd.DataFrame(rows)

        if len(self.references):
            thresholds = [self._threshold(key)[0] for key in keys]
            self.references['calibrated_threshold_ratio'] = thresholds
            self.references['calibrated_status'] = np.where(
                (
                    self.references.free_radius_A + settings.geometry_numeric_tolerance_A
                    >= self.references.probe_radius_A * self.references.calibrated_threshold_ratio
                ),
                'passed',
                'rejected',
            )


    def _threshold(self, key):
        """Выбираем порог совместимого типа, не ужесточая процентное правило."""
        settings = self.settings

        # Без подходящего типа остаётся прежний процентный критерий.
        baseline = settings.geometry_minimum_fit_fraction - settings.geometry_overlap_tolerance_percent / 100.0
        references = self.types.get(key, [])

        if not references:
            return (baseline, None)
        reference = min(references, key=lambda row: row['reference_ratio'])

        # Основные 10% не добавляем повторно к дефициту занятого эталона.
        adjusted = max(0.0, reference['reference_ratio'] - settings.geometry_reference_margin_percent / 100.0)

        return (min(baseline, adjusted), reference)


    def evaluate(self, point, data):
        settings = self.settings
        radius = data['probe_radius_A']
        issue, composition = ('', '')

        # Грубый тип точки определяем один раз.
        cache_key = tuple(np.asarray(point, dtype=float) % 1.0)

        if cache_key not in self.cache:
            try:
                self.cache[cache_key] = (*geometry_reference_type(self.structure, point, data, settings), '')
            except ValueError as error:
                self.cache[cache_key] = (None, '', str(error))
        key, composition, issue = self.cache[cache_key]
        threshold, reference = self._threshold(key)
        baseline = settings.geometry_minimum_fit_fraction - settings.geometry_overlap_tolerance_percent / 100.0
        applied = threshold < baseline

        # Меняем только геометрическое решение; исходные зазоры сохраняем.
        margin = data['free_radius_A'] - threshold * radius
        status = 'passed' if margin >= -settings.geometry_numeric_tolerance_A else 'rejected'
        reason = data['geometry_reason']

        if status == 'passed' and data['geometry_status'] == 'rejected':
            reason = 'fits_occupied_site_reference'

        return dict(
            geometry_status=status,
            geometry_reason=reason,
            geometry_uncalibrated_status=data['geometry_status'],
            geometry_free_radius_ratio=data['free_radius_A'] / radius,
            geometry_reference_status='matched_type' if reference is not None else 'percentage_fallback',
            geometry_reference_composition=composition,
            geometry_reference_site_index=reference['site_index'] if reference is not None else None,
            geometry_reference_ratio=reference['reference_ratio'] if reference is not None else None,
            geometry_threshold_ratio=threshold,
            geometry_effective_minimum_free_radius_A=threshold * radius,
            geometry_effective_allowed_overlap_A=(settings.geometry_minimum_fit_fraction - threshold) * radius,
            geometry_effective_margin_A=margin,
            geometry_reference_applied=bool(applied),
            geometry_reference_issue=issue,
        )
