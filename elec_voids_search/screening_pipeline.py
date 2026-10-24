"""Notebook entry points for full screening and saved-geometry continuation."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pymatgen.core import Structure

from .config import resolve_config
from .post_geometry_config import PostGeometryConfig
from .post_geometry import screen_geometry_candidates
from .post_geometry_exports import export_post_geometry
from .run_search import run_search


def read_geometry_candidates(candidates_csv):
    saved = pd.read_csv(candidates_csv)
    if 'in_geometry' in saved:
        flags = saved.in_geometry
        if flags.isna().any() or not flags.isin([True,False]).all():
            raise ValueError('in_geometry must contain explicit Boolean values')
        return saved.loc[flags.eq(True)].copy()
    if 'geometry_status' in saved:
        return saved.loc[saved.geometry_status.eq('passed')].copy()
    raise ValueError('Saved table must contain in_geometry or geometry_status')


def continue_from_saved_geometry(structure_path,candidates_csv,oxidation_states,config=None,output_dir=None):
    """Both default branches start from geometry, regardless of old Li status."""
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    structure_path,candidates_csv = Path(structure_path),Path(candidates_csv)
    structure = Structure.from_file(structure_path)
    result = screen_geometry_candidates(structure,read_geometry_candidates(candidates_csv),oxidation_states,config)
    result['provenance'] = dict(structure_file=str(structure_path.resolve()),
        structure_sha256=hashlib.sha256(structure_path.read_bytes()).hexdigest(),
        saved_geometry_source=str(candidates_csv.resolve()),
        saved_geometry_sha256=hashlib.sha256(candidates_csv.read_bytes()).hexdigest(),previous_stages_rerun=False)
    if output_dir is not None:
        export_post_geometry(structure,result,output_dir,result['provenance'])
    return result


def run_screening(structure_path,search_config=None,post_config=None,output_dir=None):
    """Run the frozen search through geometry, then configurable filter paths.

    Search tables are returned in memory. Optional disk output contains only
    compact post-geometry results; earlier stages are not repeatedly exported.
    """
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')
    settings = resolve_config(search_config)
    final_settings = post_config or PostGeometryConfig(probe_element=settings.probe_element,
                                                       probe_oxidation_state=settings.probe_oxidation_state)
    if (final_settings.probe_element != settings.probe_element
            or final_settings.probe_oxidation_state != settings.probe_oxidation_state):
        raise ValueError('Search and post-geometry filters must use the same probe element and charge')
    with TemporaryDirectory(prefix='elec_voids_search_') as temporary:
        search = run_search(structure_path,Path(temporary)/'search',config=settings,save_results=False)
    structure = Structure.from_file(structure_path)
    final = screen_geometry_candidates(structure,search['geometry'],settings.geometry_oxidation_states,final_settings)
    final['provenance'] = dict(structure_file=str(Path(structure_path).resolve()),
                              search_settings=settings.to_dict(),previous_stages_rerun=True)
    if output_dir is not None:
        export_post_geometry(structure,final,output_dir,final['provenance'])
    return dict(search=search,post_geometry=final)
