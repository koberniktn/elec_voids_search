"""Подробный экспорт отдельного запуска ранних стадий поиска."""

import hashlib
import json
import sys

from .exports import export_candidates


def export_search_result(structure, structure_path, result, output_dir):
    """Отдельно сохраняем подробный результат поиска до геометрии."""

    from ..config import resolve_config

    settings = resolve_config(result['settings'])
    geometry_results = result
    grid, voronoi = result['grid'], result['voronoi']
    combined, merged = result['combined'], result['merged']
    nonequivalent, clusters = result['nonequivalent'], result['clusters']
    parameters, summary = result['parameters'], result['summary']
    voronoi_details = {'raw_voronoi_df': result['raw_voronoi_vertices']}

    output_dir.mkdir(parents=True, exist_ok=False)

    for name, table in [
        ('grid', grid),
        ('voronoi', voronoi),
        ('combined', combined),
        ('merged', merged),
        ('nonequivalent', nonequivalent),
        *[
            (name, geometry_results[name])
            for name in [
                'geometry_evaluated',
                'geometry',
                'geometry_indeterminate',
                'geometry_rejected',
                'geometry_retained',
            ]
        ],
    ]:
        export_candidates(
            structure,
            table,
            output_dir,
            name,
            parameters,
            write_cif=settings.export_cif,
        )
    voronoi_details['raw_voronoi_df'].to_csv(
        output_dir / 'raw_voronoi_vertices.csv',
        index=False,
    )
    clusters.to_csv(output_dir / 'candidate_clusters.csv', index=False)
    (output_dir / 'candidate_clusters.json').write_text(
        clusters.to_json(orient='records', indent=2, force_ascii=False, double_precision=15) + '\n',
    )

    for name in ['geometry_candidate_audit', 'site_radii', 'occupied_probe_controls']:
        table = geometry_results[name]
        table.to_csv(output_dir / f'{name}.csv', index=False)
        (output_dir / f'{name}.json').write_text(table.to_json(orient='records', indent=2, double_precision=15))

    if settings.geometry_reference_mode == 'occupied_sites':
        geometry_results['occupied_geometry_references'].to_csv(
            output_dir / 'occupied_geometry_references.csv',
            index=False,
        )
    summary.to_csv(output_dir / 'summary.csv', index=False)
    metadata = {
        'structure_file': str(structure_path),
        'input_sha256': hashlib.sha256(structure_path.read_bytes()).hexdigest(),
        'python': sys.executable,
        'parameters': parameters,
    }
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')

