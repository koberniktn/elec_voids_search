"""Общие таблицы полного скрининга и CIF для каждой стадии отбора."""

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

from .post_geometry_exports import write_candidate_markers


# Имена основных стадий совпадают с прежним тестовым примером.

SEARCH_STAGES = {
    '01_grid': 'grid',
    '02_voronoi': 'voronoi',
    '03_combined': 'combined',
    '04_merged': 'merged',
    '05_symmetry': 'nonequivalent',
    '06_geometry': 'geometry',
}


def screening_tables(results):
    """Собираем таблицы из одного результата или словаря результатов, без расчётов."""

    if 'search' in results and 'post_geometry' in results:
        source = results['post_geometry']['provenance']['structure_file']
        results = {Path(source).stem: results}

    if not results:
        raise ValueError('At least one screening result is required')

    summaries, steps, structures = [], [], {}

    for name, result in results.items():
        if not isinstance(name, str) or name in {'.', '..'} or '/' in name or '\\' in name or not name:
            raise ValueError('Structure names must be simple folder names')

        search, post = result['search'], result['post_geometry']
        stages = {name: search[key] for name, key in SEARCH_STAGES.items()}
        row = dict(zip(
            ['Соединение', 'Сетка', 'Вороной', 'Объединено', 'Кластеры', 'Симметрия', 'Геометрия'],
            [name] + [len(table) for table in stages.values()],
        ))
        canonical = {
            ('coordination',): ('07_coordination', 'Координация'),
            ('li_distance',): ('08_li_distance', 'Li–Li'),
            ('coordination', 'li_distance'): ('09_both', 'Коорд. → Li–Li'),
            ('li_distance', 'coordination'): ('09_both', 'Li–Li → коорд.'),
        }

        # Имена маршрутов пользователь может менять; смысл задаёт последовательность.

        for branch_name, branch in post['branches'].items():
            sequence = tuple(post['settings']['paths'][branch_name])
            stage, title = canonical.get(sequence, (f'branch_{branch_name}', branch_name))
            table = branch['passed']
            if stage in stages and set(stages[stage].candidate_id) != set(table.candidate_id):
                stage = f'branch_{branch_name}'
            stages[stage] = table
            row[title] = len(table)

        candidates = search['combined'].drop(
            columns=['merge_member_candidate_ids', 'merge_source_methods'], errors='ignore',
        ).copy()
        for stage, table in stages.items():
            flag = stage[3:] if stage[:2].isdigit() else stage
            candidates[f'in_{flag}'] = candidates.candidate_id.isin(table.candidate_id)
        for table in (search['geometry_evaluated'], post['candidates']):
            columns = ['candidate_id'] + [key for key in table if key not in candidates.columns]
            candidates = candidates.merge(table[columns], on='candidate_id', how='left', validate='one_to_one')

        branch_steps = post['summary'].copy()
        branch_steps.insert(0, 'structure', name)
        steps.append(branch_steps)
        summaries.append(row)
        structures[name] = dict(stages=stages, candidates=candidates, result=result)

    return dict(
        results=results,
        summary=pd.DataFrame(summaries),
        branch_steps=pd.concat(steps, ignore_index=True),
        structures=structures,
    )


def export_screening_results(results, output_dir, export_cif=True, run_notes=None):
    """Сохраняем один или несколько результатов; существующую папку не перезаписываем."""

    report = screening_tables(results)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), structures={})
    metadata['versions'] = {}
    for package in ('numpy', 'scipy', 'pandas', 'pymatgen', 'gridmlip'):
        try:
            metadata['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata['versions'][package] = None
    metadata['search_source_sha256'] = {
        str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(__file__).resolve().parents[1].rglob('*.py')
    }

    for name, item in report['structures'].items():
        folder = output / name
        folder.mkdir()
        result = item['result']
        search, post = result['search'], result['post_geometry']
        source = Path(post['provenance']['structure_file'])
        item['candidates'].to_csv(folder / 'candidates.csv', index=False)
        for key, filename in [
            ('site_radii', 'site_radii'),
            ('occupied_probe_controls', 'occupied_geometry_controls'),
            ('occupied_geometry_references', 'occupied_geometry_references'),
        ]:
            if key in search and (key != 'occupied_geometry_references' or len(search[key])):
                search[key].to_csv(folder / f'{filename}.csv', index=False)
        if 'occupied_reference_descriptors' in post:
            post['occupied_reference_descriptors'].to_csv(folder / 'occupied_descriptors.csv', index=False)

        if export_cif:
            host = result.get('structure')
            if host is None:
                host = Structure.from_file(source)
            (folder / 'cif').mkdir()
            for stage, table in item['stages'].items():
                write_candidate_markers(host, table, folder / 'cif' / f'{stage}.cif')

        metadata['structures'][name] = dict(
            provenance=post['provenance'],
            input_sha256=(post['provenance']['structure_sha256']
                          if 'structure_sha256' in post['provenance']
                          else hashlib.sha256(source.read_bytes()).hexdigest()),
            post_geometry_settings=post['settings'],
            run_notes=(run_notes or {}).get(name, {}),
        )

    report['summary'].to_csv(output / 'summary.csv', index=False)
    report['branch_steps'].to_csv(output / 'branch_steps.csv', index=False)
    (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    report['output_dir'] = output

    return report
