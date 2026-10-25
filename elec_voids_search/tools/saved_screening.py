"""Загрузка сохранённой геометрии и отдельное продолжение финального отбора."""

import hashlib
from pathlib import Path

import pandas as pd
from pymatgen.core import Structure

from ..filtering.post_geometry import screen_geometry_candidates
from ..results.post_geometry_exports import export_post_geometry


def read_geometry_candidates(candidates_csv):
    """Читаем только геометрически прошедшие точки из сохранённой таблицы."""
    saved = pd.read_csv(candidates_csv)

    if 'in_geometry' in saved:
        flags = saved.in_geometry

        if flags.isna().any() or not flags.isin([True, False]).all():
            raise ValueError('in_geometry must contain explicit Boolean values')

        return saved.loc[flags.eq(True)].copy()

    if 'geometry_status' in saved:
        return saved.loc[saved.geometry_status.eq('passed')].copy()
    raise ValueError('Saved table must contain in_geometry or geometry_status')



def continue_from_saved_geometry(
    structure_path,
    candidates_csv,
    oxidation_states,
    config=None,
    output_dir=None,
):
    """Продолжаем с геометрии независимо от прежнего решения Li–Li."""

    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    structure_path, candidates_csv = (Path(structure_path), Path(candidates_csv))
    structure = Structure.from_file(structure_path)

    # Прежний результат Li–Li не ограничивает вход в новые ветки.
    result = screen_geometry_candidates(
        structure,
        read_geometry_candidates(candidates_csv),
        oxidation_states,
        config,
    )
    result['provenance'] = dict(
        structure_file=str(structure_path.resolve()),
        structure_sha256=hashlib.sha256(structure_path.read_bytes()).hexdigest(),
        saved_geometry_source=str(candidates_csv.resolve()),
        saved_geometry_sha256=hashlib.sha256(candidates_csv.read_bytes()).hexdigest(),
        previous_stages_rerun=False,
    )

    if output_dir is not None:
        export_post_geometry(structure, result, output_dir, result['provenance'])

    return result

