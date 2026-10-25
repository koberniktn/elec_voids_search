"""Совместимый прежний запуск только Li–Li-фильтра из сохранённой геометрии."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from pymatgen.core import DummySpecies, Structure
from pymatgen.io.cif import CifWriter

from ..filtering.probe_separation import screen_probe_separation


def continue_screening_from_saved(structure_path, candidates_csv, output_dir=None, config=None):
    """Продолжаем только Li–Li-отбор без повторного поиска полостей.

    При заданной папке сохраняем CSV, занятые сайты, параметры и CIF двух групп.
    """
    structure_path, candidates_csv = (Path(structure_path), Path(candidates_csv))

    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    saved = pd.read_csv(candidates_csv)
    geometry = saved.loc[saved.in_geometry.eq(True)].copy()
    structure = Structure.from_file(structure_path)
    result = screen_probe_separation(structure, geometry, config=config)
    result['summary'].insert(0, 'structure_file', structure_path.name)

    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=False)
        result['probe_distance_evaluated'].to_csv(folder / 'candidates.csv', index=False)

        for name in ['occupied_probe_sites', 'occupied_probe_nonequivalent']:
            result[name].to_csv(folder / f'{name}.csv', index=False)

        for key, filename in [
            ('probe_distance_passed', '07_probe_distance_passed.cif'),
            ('probe_distance_rejected', 'removed_04_probe_distance.cif'),
        ]:
            host = structure.copy()

            for point in result[key][['frac_x', 'frac_y', 'frac_z']].to_numpy():
                host.append(DummySpecies('X'), point)
            CifWriter(host, symprec=None).write_file(folder / filename)
        metadata = {
            'settings': result['settings'],
            'reference_statistic': 'mean_over_all_occupied_sites_of_nearest_other_periodic_probe_distance',
            'pass_rule': 'nearest_distance + distance_tolerance_A + numeric_tolerance_A >= mean_distance',
            'structure_file': str(structure_path.resolve()),
            'structure_sha256': hashlib.sha256(structure_path.read_bytes()).hexdigest(),
            'saved_geometry_source': str(candidates_csv.resolve()),
            'saved_geometry_source_sha256': hashlib.sha256(candidates_csv.read_bytes()).hexdigest(),
            'search_and_geometry_rerun': False,
        }
        (folder / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

    return result
