"""Исследовательское сравнение первых сфер с выравниванием координат."""

from itertools import permutations, product
import math

import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ..symmetry.occupied_sites import get_nonequivalent_occupied_sites
from ..descriptors.coordination_shell import (
    EnvironmentMatchConfig,
    describe_framework_shell,
    compare_shell_geometry,
)


def compare_framework_shells(candidate, reference, config=None):
    """Сравниваем сферы и совмещаем их поворотом без сдвига и масштабирования."""
    settings = config or EnvironmentMatchConfig()
    result = compare_shell_geometry(candidate, reference, settings)

    if not result['similar']:
        return result
    result.update(similar=False, comparison_status='rejected', reason='bond_lengths_or_angles')
    labels = candidate['labels']
    ratios = [
        result['radial_rms_A'] / settings.radial_rms_tolerance_A,
        result['angle_rms_deg'] / settings.angle_rms_tolerance_deg,
    ]

    # Точный перебор перестановок ограничен явным вычислительным бюджетом.
    groups = [
        [i for i, symbol in enumerate(reference['labels']) if symbol == element]
        for element in sorted(set(labels))
    ]
    total = math.prod((math.factorial(len(group)) for group in groups))

    if total > settings.maximum_exact_permutations:
        result.update(comparison_status='indeterminate', reason='permutation_budget')

        return result
    orders = np.asarray(
        [
            sum((list(part) for part in combination), [])
            for combination in product(*(permutations(group) for group in groups))
        ],
        dtype=int,
    )
    left = candidate['vectors']
    right = reference['vectors'][orders]

    # Совмещаем окружения без подгонки сдвига и масштаба.
    covariance = np.einsum('ni,pnj->pij', left, right)
    u, _, vt = np.linalg.svd(covariance)
    rotations = u @ vt

    if not settings.allow_reflection:
        correction = np.ones((len(rotations), 3))
        correction[:, -1] = np.linalg.det(rotations)
        rotations = u * correction[:, None, :] @ vt
    transformed = np.einsum('ni,pij->pnj', left, rotations)
    aligned = float(np.sqrt(np.min(np.mean(np.sum((transformed - right) ** 2, axis=2), axis=1))))
    ratios.append(aligned / settings.aligned_rms_tolerance_A)
    passed = max(ratios) <= 1.0 + settings.numeric_tolerance
    result.update(
        similar=passed,
        reason='similar_shell' if passed else 'aligned_coordinates',
        aligned_rms_A=aligned,
        normalized_error=max(ratios),
    )
    result['comparison_status'] = 'passed' if passed else 'rejected'

    return result


class FrameworkEnvironmentMatcher:
    """Прежнее исследовательское сравнение первых сфер с выравниванием."""


    def __init__(
        self,
        structure,
        oxidation_states,
        probe_element='Li',
        probe_oxidation_state=1,
        config=None,
        reference_indices=None,
    ):
        if not structure.is_ordered:
            raise ValueError('Resolve partial occupancies before environment comparison')
        self.structure = structure
        self.oxidation_states = dict(oxidation_states)
        self.probe_element = probe_element
        self.probe_oxidation_state = probe_oxidation_state
        self.config = config or EnvironmentMatchConfig()
        missing = {site.specie.symbol for site in structure} - set(oxidation_states)

        if missing:
            raise ValueError(f'Missing manual oxidation states: {sorted(missing)}')

        if reference_indices is None:
            analyzer = SpacegroupAnalyzer(structure, symprec=self.config.symmetry_symprec_A)
            groups = get_nonequivalent_occupied_sites(structure, analyzer, probe_element)

            if groups.empty:
                raise ValueError('No occupied reference sites')
            reference_indices = groups.atom_indices.map(lambda group: group[0]).tolist()
        self.references = {
            int(index): self.describe(structure[int(index)].frac_coords)
            for index in reference_indices
        }

        if not self.references:
            raise ValueError('No occupied reference sites')


    def describe(self, point):
        return describe_framework_shell(
            self.structure,
            point,
            self.oxidation_states,
            self.probe_element,
            self.probe_oxidation_state,
            self.config,
        )


    def screen(self, candidates):
        rows = []

        for _, candidate in candidates.iterrows():
            shell = self.describe(candidate[['frac_x', 'frac_y', 'frac_z']].to_numpy(dtype=float))
            pairs = [
                (index, compare_framework_shells(shell, reference, self.config))
                for (index, reference) in self.references.items()
            ]
            passing = [pair for pair in pairs if pair[1]['similar']]
            index, best = min(
                passing or pairs,
                key=lambda pair: (
                    pair[1]['normalized_error'] is None,
                    pair[1]['normalized_error'] if pair[1]['normalized_error'] is not None else np.inf,
                ),
            )
            row = candidate.to_dict()
            unresolved = any((pair[1]['comparison_status'] == 'indeterminate' for pair in pairs))
            row.update(
                environment_similar=bool(passing),
                environment_status='passed' if passing else 'indeterminate' if unresolved else 'rejected',
                environment_reference_site_index=index,
                environment_similar_site_indices=[pair[0] for pair in passing],
                environment_reason=best['reason'],
                environment_coordination=shell['coordination'],
                environment_shell_composition=str(shell['composition']),
                environment_radial_rms_A=best['radial_rms_A'],
                environment_angle_rms_deg=best['angle_rms_deg'],
                environment_aligned_rms_A=best['aligned_rms_A'],
            )
            rows.append(row)
        extra = [
            'environment_similar',
            'environment_status',
            'environment_reference_site_index',
            'environment_similar_site_indices',
            'environment_reason',
            'environment_coordination',
            'environment_shell_composition',
            'environment_radial_rms_A',
            'environment_angle_rms_deg',
            'environment_aligned_rms_A',
        ]

        return pd.DataFrame(rows, columns=list(candidates.columns) + extra)


def screen_environment_similarity(
    structure,
    candidates,
    oxidation_states,
    probe_element='Li',
    probe_oxidation_state=1,
    config=None,
):
    """Применяем исследовательское сравнение к готовым кандидатам.

    Ранние стадии не запускаем, ID и координаты сохраняем.
    """
    matcher = FrameworkEnvironmentMatcher(
        structure,
        oxidation_states,
        probe_element,
        probe_oxidation_state,
        config=config,
    )
    evaluated = matcher.screen(candidates)

    return {
        'environment_evaluated': evaluated,
        'environment_passed': evaluated.loc[evaluated.environment_status.eq('passed')].copy(),
        'environment_rejected': evaluated.loc[evaluated.environment_status.eq('rejected')].copy(),
        'environment_indeterminate': evaluated.loc[evaluated.environment_status.eq('indeterminate')].copy(),
        'settings': matcher.config.to_dict(),
    }
