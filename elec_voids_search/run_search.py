"""Отдельный запуск ранних стадий поиска, до геометрии включительно."""

from pathlib import Path

from pymatgen.core import Structure

from .config import resolve_config
from .geometry.geometry import screen_geometry
from .geometry.ionic_geometry import validate_oxidation_states
from .candidates.search_stages import find_candidate_sites, assemble_search_result
from .results.search_exports import export_search_result


def run_search(structure_path, output_dir=None, config=None, save_results=True, **overrides):
    """Поиск до геометрии; полный скрининг выполняет run_screening."""

    settings = resolve_config(config, **overrides)
    structure_path = Path(structure_path).resolve()

    if save_results:
        if output_dir is None:
            raise ValueError('output_dir is required when save_results=True')
        output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError(f'Output directory already exists: {output_dir}')

    structure = Structure.from_file(structure_path)
    validate_oxidation_states(structure, settings)

    sites = find_candidate_sites(structure, structure_path, settings)
    geometry = screen_geometry(structure, sites['nonequivalent'], settings)
    result = assemble_search_result(structure, structure_path, settings, sites, geometry)

    if save_results:
        export_search_result(structure, structure_path, result, output_dir)

    return result
