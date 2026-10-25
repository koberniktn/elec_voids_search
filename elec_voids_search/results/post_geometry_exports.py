"""Компактное сохранение таблиц, настроек и CIF с маркерами полостей."""

import json
from pathlib import Path

from pymatgen.core import DummySpecies
from pymatgen.io.cif import CifWriter


def write_candidate_markers(structure, table, path):
    """Записываем исходный каркас и X-маркеры выбранных полостей в CIF."""
    host = structure.copy()

    for point in table[['frac_x', 'frac_y', 'frac_z']].to_numpy():
        host.append(DummySpecies('X'), point)
    CifWriter(host, symprec=None).write_file(path)


def export_post_geometry(structure, result, output_dir, provenance=None):
    """Сохраняем итоговые таблицы и по запросу CIF; существующую папку не затираем."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    result['candidates'].to_csv(folder / 'candidates.csv', index=False)
    result['summary'].to_csv(folder / 'summary.csv', index=False)

    for key in ['occupied_reference_descriptors', 'occupied_probe_sites', 'occupied_probe_nonequivalent']:
        if key in result:
            result[key].to_csv(folder / f'{key}.csv', index=False)
    metadata = dict(
        settings=result['settings'],
        provenance=provenance or {},
        input_stage='geometry_passed',
        candidate_coordinates='unchanged',
    )
    (folder / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

    if result['settings']['export_cif']:
        for name, branch in result['branches'].items():
            for position, step in enumerate(branch['steps'], 1):
                stem = f'{name}_{position:02d}_{step['filter']}'
                write_candidate_markers(structure, step['passed'], folder / f'{stem}_passed.cif')

                if result['settings']['export_rejected_cif']:
                    write_candidate_markers(
                        structure,
                        step['rejected'],
                        folder / f'{stem}_rejected.cif',
                    )

                    if len(step['indeterminate']):
                        write_candidate_markers(
                            structure,
                            step['indeterminate'],
                            folder / f'{stem}_indeterminate.cif',
                        )

    return folder
