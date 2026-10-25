"""Запуск полного пайплайна"""

import hashlib
from pathlib import Path

from pymatgen.core import Structure

from .config import resolve_config
from .filtering.post_geometry_config import PostGeometryConfig
from .filtering.post_geometry import screen_geometry_candidates
from .geometry.geometry import screen_geometry
from .geometry.ionic_geometry import validate_oxidation_states
from .candidates.search_stages import find_candidate_sites, assemble_search_result
from .results.screening_results import export_screening_results


def run_screening(
    structure_path,
    search_config=None,
    post_config=None,
    output_dir=None,
    export_cif=True,
):
    """Единый полный скрининг одной структуры.

    Сетка и Вороной → объединение → кластеры → симметрия → геометрия
    → заданные независимые или последовательные ветки координации и Li–Li.
    Сохраняем результаты только после завершения всех этапов.
    """

    structure_path = Path(structure_path).resolve()
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')

    settings = resolve_config(search_config)
    final_settings = post_config if post_config is not None else PostGeometryConfig(
        probe_element=settings.probe_element,
        probe_oxidation_state=settings.probe_oxidation_state,
        paths={
            'coordination': ('coordination',),
            'li_distance': ('li_distance',),
            'coord_then_li': ('coordination', 'li_distance'),
            'li_then_coord': ('li_distance', 'coordination'),
        },
    )
    if not isinstance(final_settings, PostGeometryConfig):
        raise TypeError('post_config must be PostGeometryConfig or None')
    if (
        final_settings.probe_element != settings.probe_element
        or final_settings.probe_oxidation_state != settings.probe_oxidation_state
    ):
        raise ValueError('Search and post-geometry filters must use the same probe element and charge')


    # Чтение структуры.

    structure = Structure.from_file(structure_path)
    validate_oxidation_states(structure, settings)

    # 1–4. Два сырых поиска, общая кластеризация и отсев по симметрии.

    sites = find_candidate_sites(structure, structure_path, settings)

    # 5. Геометрия получает только симметрийных представителей.

    geometry = screen_geometry(structure, sites['nonequivalent'], settings)
    search = assemble_search_result(structure, structure_path, settings, sites, geometry)

    # 6. Финальные этапы скрининга - Li-Li критерий и критерий координационного окружения

    final = screen_geometry_candidates(
        structure, geometry['geometry'], settings.geometry_oxidation_states, final_settings,
    )
    final['provenance'] = dict(
        structure_file=str(structure_path),
        structure_sha256=hashlib.sha256(structure_path.read_bytes()).hexdigest(),
        search_settings=settings.to_dict(),
        previous_stages_rerun=True,
    )
    result = dict(structure=structure, search=search, post_geometry=final)

    # Экспорт использует готовые стадии, ни один фильтр здесь не запускается.

    if output_dir is not None:
        export_screening_results(result, output_dir, export_cif=export_cif)

    return result
