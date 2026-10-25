"""Необязательная диагностика известных вакансий; в отборе не участвует."""

import pandas as pd
from pymatgen.core import Structure

from ..results.screening_results import screening_tables


def reference_recovery(results, targets, tolerance_A=0.25):
    """Диагностика известных вакансий: targets = {имя: {метка: дробные координаты}}."""

    import math
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    if not math.isfinite(tolerance_A) or tolerance_A < 0:
        raise ValueError('tolerance_A must be finite and nonnegative')
    report = screening_tables(results)
    rows = []
    for name, points in targets.items():
        item = report['structures'][name]
        result = item['result']
        settings = result['search']['settings']
        host = result.get('structure')
        if host is None:
            host = Structure.from_file(result['post_geometry']['provenance']['structure_file'])
        operations = SpacegroupAnalyzer(
            host, symprec=settings['symmetry_symprec_A'],
            angle_tolerance=settings['symmetry_angle_tolerance_deg'],
        ).get_symmetry_operations()
        for label, point in points.items():
            orbit = [operation.operate(point) % 1 for operation in operations]
            for stage, table in item['stages'].items():
                error = float(host.lattice.get_all_distances(
                    orbit, table[['frac_x', 'frac_y', 'frac_z']].to_numpy(dtype=float),
                ).min()) if len(table) else float('nan')
                rows.append(dict(
                    structure=name, target=label, stage=stage,
                    error_A=error, recovered=bool(math.isfinite(error) and error <= tolerance_A),
                ))

    return pd.DataFrame(rows, columns=['structure', 'target', 'stage', 'error_A', 'recovered'])

