"""Удобный запуск одинакового пайплайна для одного CIF или набора структур."""

from collections.abc import Mapping
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
from pathlib import Path
from time import perf_counter
import warnings

from .config import resolve_config
from .screening_pipeline import run_screening
from .results.screening_results import export_screening_results, screening_tables


def run_structures(
    files,
    oxidation_states=None,
    search_config=None,
    post_config=None,
    output_dir=None,
    export_cif=True,
    quiet=True,
):
    """Принимаем путь, список путей или словарь {имя: путь}.

    oxidation_states — общий словарь зарядов либо {имя: словарь зарядов}.
    Без output_dir результаты и итоговые таблицы остаются только в памяти.
    """

    if isinstance(files, (str, Path)):
        files = [files]
    if isinstance(files, Mapping):
        paths = {name: Path(path) for name, path in files.items()}
    else:
        paths = {}
        for value in files:
            path = Path(value)
            if path.stem in paths:
                raise ValueError('Duplicate file stems; use a mapping with explicit structure names')
            paths[path.stem] = path
    if not paths:
        raise ValueError('At least one structure is required')
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')

    settings = resolve_config(search_config)
    configured = {}

    # Готовим все входы до начала поиска.

    for name, path in paths.items():
        if not isinstance(name, str) or not name or name in {'.', '..'} or '/' in name or '\\' in name:
            raise ValueError('Structure names must be simple folder names')
        if not path.is_file():
            raise FileNotFoundError(path)
        charges = oxidation_states
        if charges is not None and any(isinstance(value, Mapping) for value in charges.values()):
            charges = charges[name]
        configured[name] = settings if charges is None else replace(settings, geometry_oxidation_states=charges)

    results, notes = {}, {}
    for name, path in paths.items():
        start = perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            if quiet:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    result = run_screening(path, search_config=configured[name], post_config=post_config)
            else:
                result = run_screening(path, search_config=configured[name], post_config=post_config)
        results[name] = result
        notes[name] = dict(
            seconds=round(perf_counter() - start, 2),
            warnings=sorted({str(item.message) for item in captured}),
        )

    report = (
        screening_tables(results) if output_dir is None
        else export_screening_results(results, output_dir, export_cif=export_cif, run_notes=notes)
    )
    report['run_notes'] = notes

    return report
