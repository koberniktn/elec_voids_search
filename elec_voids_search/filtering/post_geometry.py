"""Независимые и последовательные ветки отбора после геометрии."""

import numpy as np
import pandas as pd

from .post_geometry_config import PostGeometryConfig
from .probe_separation import screen_probe_separation
from .weighted_voronoi import WeightedVoronoiMatcher


class _FilterSession:
    """Храним оценки фильтров по ID кандидата для всех веток одного запуска."""


    def __init__(self, structure, oxidation_states, settings):
        self.structure = structure
        self.oxidation_states = oxidation_states
        self.settings = settings
        self.matcher = None
        self.evaluations = {'coordination': {}, 'li_distance': {}}
        self.auxiliary = {}


    def evaluate(self, key, table):
        cache = self.evaluations[key]

        # Считаем только кандидатов, ещё не оценённых этим фильтром.
        missing = table.loc[~table.candidate_id.isin(cache)].copy()

        if len(missing):
            if key == 'coordination':
                if self.matcher is None:
                    self.matcher = WeightedVoronoiMatcher(
                        self.structure,
                        self.oxidation_states,
                        self.settings.probe_element,
                        self.settings.probe_oxidation_state,
                        self.settings.coordination,
                    )
                    self.auxiliary['occupied_reference_descriptors'] = self.matcher.reference_table
                evaluated = self.matcher.screen(missing)
                columns = [name for name in evaluated if name.startswith('coordination_')]
            else:
                result = screen_probe_separation(self.structure, missing, self.settings.resolved_separation)
                evaluated = result['probe_distance_evaluated']
                columns = [
                    name
                    for name in evaluated
                    if (
                        (name not in missing.columns)
                        or (name in {
                            'probe_distance_status',
                            'screening_group',
                            'separation_probe_element',
                            'nearest_occupied_probe_site_index',
                            'nearest_occupied_probe_distance_A',
                            'mean_probe_nearest_neighbor_A',
                            'probe_distance_tolerance_A',
                            'probe_distance_margin_A',
                        })
                    )
                ]
                self.auxiliary.update(
                    {
                        name: result[name]
                        for name in ['occupied_probe_sites', 'occupied_probe_nonequivalent']
                    },
                )

            for _, row in evaluated.iterrows():
                cache[row.candidate_id] = {name: row[name] for name in columns}

        # Для остальных веток восстанавливаем оценки из общего кэша.
        output = table.copy()
        columns = sorted({column for values in cache.values() for column in values})
        status = 'coordination_status' if key == 'coordination' else 'probe_distance_status'

        if status not in columns:
            columns.append(status)

        for column in columns:
            output[column] = [cache[cid].get(column) for cid in table.candidate_id]

        return output


def screen_geometry_candidates(structure, geometry_candidates, oxidation_states, config=None):
    """Запускаем заданные ветки из общей геометрической выборки.

    Внутри ветки следующий фильтр получает только прошедшие предыдущий.
    Оценки по ID общие для веток; научные расчёты не повторяются.
    """
    settings = config or PostGeometryConfig()

    if not isinstance(settings, PostGeometryConfig):
        raise TypeError('config must be PostGeometryConfig')
    required = {'candidate_id', 'frac_x', 'frac_y', 'frac_z'}
    missing = required - set(geometry_candidates.columns)

    if missing:
        raise ValueError(f'Missing candidate columns: {sorted(missing)}')

    if not geometry_candidates.candidate_id.is_unique:
        raise ValueError('Candidate IDs must be unique in the supplied structure')

    if geometry_candidates.candidate_id.isna().any():
        raise ValueError('Candidate IDs must not be missing')

    if not np.isfinite(geometry_candidates[['frac_x', 'frac_y', 'frac_z']].to_numpy(dtype=float)).all():
        raise ValueError('Candidate coordinates must be finite')

    if (
        ('geometry_status' in geometry_candidates)
        and (not geometry_candidates.geometry_status.eq('passed').all())
    ):
        raise ValueError('Input must contain only geometrically passed candidates')

    # Все ветки используют общий кэш и одну геометрическую выборку.
    session = _FilterSession(structure, dict(oxidation_states), settings)
    branches, summary = ({}, [])
    combined = geometry_candidates.copy()

    # Каждая независимая ветка начинается с геометрии.
    for name, path in settings.paths.items():
        current = geometry_candidates.copy()
        steps, rejected_parts, indeterminate_parts = ([], [], [])
        branch_status = pd.Series('passed', index=geometry_candidates.candidate_id, dtype=object)

        for position, key in enumerate(path, 1):
            evaluated = session.evaluate(key, current)
            column = 'coordination_status' if key == 'coordination' else 'probe_distance_status'
            passed = evaluated[evaluated[column].eq('passed')].copy()
            rejected = evaluated[evaluated[column].eq('rejected')].copy()
            indeterminate = evaluated[evaluated[column].eq('indeterminate')].copy()
            branch_status.loc[rejected.candidate_id] = 'rejected'
            branch_status.loc[indeterminate.candidate_id] = 'indeterminate'

            for table, target in [(rejected, rejected_parts), (indeterminate, indeterminate_parts)]:
                part = table.copy()
                part['stopped_at_filter'] = key
                part['stopped_at_step'] = position
                target.append(part)
            steps.append(
                dict(
                    filter=key,
                    evaluated=evaluated,
                    passed=passed,
                    rejected=rejected,
                    indeterminate=indeterminate,
                ),
            )
            summary.append(
                dict(
                    path=name,
                    step=position,
                    filter=key,
                    n_input=len(current),
                    n_passed=len(passed),
                    n_rejected=len(rejected),
                    n_indeterminate=len(indeterminate),
                ),
            )
            current = passed
        branches[name] = dict(
            filters=tuple(path),
            steps=steps,
            passed=current,
            rejected=pd.concat(rejected_parts, ignore_index=True),
            indeterminate=pd.concat(indeterminate_parts, ignore_index=True),
        )
        combined[f'branch_{name}_status'] = combined.candidate_id.map(branch_status)
        stopped = pd.concat(rejected_parts + indeterminate_parts, ignore_index=True)
        stopped_by_id = stopped.set_index('candidate_id').stopped_at_filter
        combined[f'branch_{name}_stopped_at_filter'] = combined.candidate_id.map(stopped_by_id).fillna('')

    # Пропущенный фильтр не считается отказом: статус not_evaluated.
    for key, cache in session.evaluations.items():
        columns = sorted({column for values in cache.values() for column in values})
        status = 'coordination_status' if key == 'coordination' else 'probe_distance_status'

        for column in columns:
            combined[column] = [cache.get(cid, {}).get(column) for cid in combined.candidate_id]

        if not cache:
            combined[status] = 'not_evaluated'
        combined[status] = combined[status].fillna('not_evaluated')


    def group(row):
        a, b = (row.coordination_status, row.probe_distance_status)

        if 'not_evaluated' in {a, b}:
            return 'not_compared_by_both'

        if 'indeterminate' in {a, b}:
            return 'indeterminate'

        if a == 'passed' and b == 'passed':
            return 'passed_both'

        if a == 'passed':
            return 'coordination_only'

        if b == 'passed':
            return 'li_distance_only'

        return 'rejected_both'
    combined['filter_comparison_group'] = (
        combined.apply(group, axis=1)
        if len(combined)
        else pd.Series(index=combined.index, dtype=object)
    )

    return dict(
        candidates=combined,
        branches=branches,
        summary=pd.DataFrame(summary),
        settings=settings.to_dict(),
        **session.auxiliary,
    )
