"""Refresh coordination and intersections from saved geometry and Li–Li results."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Structure, DummySpecies
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from elec_voids_search.weighted_voronoi import WeightedVoronoiMatcher
from elec_voids_search.voronoi_coordination_config import VoronoiCoordinationConfig
from elec_voids_search.post_geometry_exports import write_candidate_markers


def refresh_coordination(example=None, config=None):
    example = Path(example) if example else Path(__file__).resolve().parent
    output = example / 'voids_gallery'
    metadata = json.loads((output / 'metadata.json').read_text())
    summary = pd.read_csv(output / 'summary.csv')
    steps = pd.read_csv(output / 'branch_steps.csv')
    recovery = pd.read_csv(output / 'reference_recovery.csv')
    config = config or VoronoiCoordinationConfig()
    preserved = {p:hashlib.sha256(p.read_bytes()).hexdigest()
                 for name in metadata['structures'] for p in (output / name).rglob('*')
                 if p.is_file() and (p.stem[:2] in {'01','02','03','04','05','06','08'}
                    or p.name in {'site_radii.csv','occupied_geometry_controls.csv'})}
    refreshed = {}
    # Complete all scientific comparisons before replacing saved outputs.
    for name, previous in metadata['structures'].items():
        table = pd.read_csv(output / name / 'candidates.csv')
        host = Structure.from_file(example / 'inputs' / f'{name}.cif')
        geometry = table.loc[table.in_geometry].copy()
        if len(geometry):
            matcher = WeightedVoronoiMatcher(host,previous['settings']['geometry_oxidation_states'],
                previous['settings']['probe_element'],previous['settings']['probe_oxidation_state'],config)
            evaluated = matcher.screen(geometry).set_index('candidate_id')
            for column in evaluated:
                if column.startswith('coordination_'):
                    table[column] = table.candidate_id.map(evaluated[column])
            references = matcher.reference_table
        else:
            references = None
        table['in_coordination'] = table.coordination_status.eq('passed')
        table['in_both'] = table.in_coordination & table.in_li_distance
        # Derive routes from the newly evaluated coordination and unchanged Li–Li decisions.
        routes = metadata['post_geometry_settings']['paths']
        for route, filters in routes.items():
            if route=='li_distance':
                continue
            active = table.in_geometry.copy()
            for position,key in enumerate(filters,1):
                column = 'coordination_status' if key=='coordination' else 'probe_distance_status'
                status = table[column]
                stopped = active & ~status.eq('passed')
                if position==1:
                    table.loc[active,f'branch_{route}_status'] = 'passed'
                    table.loc[active,f'branch_{route}_stopped_at_filter'] = ''
                table.loc[stopped,f'branch_{route}_status'] = status[stopped]
                table.loc[stopped,f'branch_{route}_stopped_at_filter'] = key
                row = steps.structure.eq(name) & steps.path.eq(route) & steps.step.eq(position)
                steps.loc[row,['n_input','n_passed','n_rejected','n_indeterminate']] = [
                    int(active.sum()),int((active & status.eq('passed')).sum()),
                    int((active & status.eq('rejected')).sum()),int((active & status.eq('indeterminate')).sum())]
                active &= status.eq('passed')
        a,b = table.coordination_status,table.probe_distance_status
        table.loc[table.in_geometry,'filter_comparison_group'] = np.select([
            (a.eq('indeterminate') | b.eq('indeterminate'))[table.in_geometry],
            (a.eq('passed') & b.eq('passed'))[table.in_geometry],
            a.eq('passed')[table.in_geometry],b.eq('passed')[table.in_geometry]],
            ['indeterminate','passed_both','coordination_only','li_distance_only'],default='rejected_both')
        row = summary['Соединение'].eq(name)
        summary.loc[row,['Координация','Коорд. → Li–Li','Li–Li → коорд.']] = [
            int(table.in_coordination.sum()),int(table.in_both.sum()),int(table.in_both.sum())]
        targets = recovery.loc[recovery.structure.eq(name),'target'].unique()
        if len(targets):
            vacancy_targets = json.loads((example/'inputs/vacancy_targets.json').read_text())
            markers = pd.read_csv(example/'inputs/LATP_reference_sites.csv').set_index('source_atom_number_1based')
            settings = previous['settings']
            operations = SpacegroupAnalyzer(host,symprec=settings['symmetry_symprec_A'],
                angle_tolerance=settings['symmetry_angle_tolerance_deg']).get_symmetry_operations()
            for target in targets:
                point = vacancy_targets[name] if target=='removed_Li' else markers.loc[int(target.split('_')[1]),['frac_x','frac_y','frac_z']].to_numpy(dtype=float)
                orbit = [operation.operate(point)%1 for operation in operations]
                for stage, flag in [('coordination','in_coordination'),('coord_then_li','in_both'),('li_then_coord','in_both')]:
                    selected = table.loc[table[flag],['frac_x','frac_y','frac_z']].to_numpy(dtype=float)
                    error = float(host.lattice.get_all_distances(orbit,selected).min()) if len(selected) else np.nan
                    match = recovery.structure.eq(name) & recovery.target.eq(target) & recovery.stage.eq(stage)
                    recovery.loc[match,['error_A','recovered']] = [error,bool(np.isfinite(error) and error<=.25)]
        refreshed[name] = (host,table,references)
        print(f'{name}: coordination={int(table.in_coordination.sum())}, both={int(table.in_both.sum())}',flush=True)
    for name,(host,table,references) in refreshed.items():
        folder = output/name
        table.to_csv(folder/'candidates.csv',index=False)
        if references is not None:references.to_csv(folder/'occupied_descriptors.csv',index=False)
        for stem,flag in [('07_coordination','in_coordination'),('09_both','in_both')]:
            selected = table.loc[table[flag]]
            write_candidate_markers(host,selected,folder/'cif'/f'{stem}.cif')
            marked = host.copy()
            for point in selected[['frac_x','frac_y','frac_z']].to_numpy(dtype=float):
                marked.append(DummySpecies('X'),point)
            Poscar(marked,comment=f'{name}: {stem}; X = candidate markers').write_file(folder/'vasp'/f'{stem}.vasp')
    summary.to_csv(output/'summary.csv',index=False)
    steps.to_csv(output/'branch_steps.csv',index=False)
    recovery.to_csv(output/'reference_recovery.csv',index=False)
    metadata['post_geometry_settings']['coordination'] = config.to_dict()
    metadata['coordination_refresh'] = {
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'input_stage':'saved_geometry_passed','previous_stages_rerun':False,'li_distance_rerun':False,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'elec_voids_search').glob('*.py')},
        'preserved_files':len(preserved),
    }
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in preserved.items())
    metadata['coordination_refresh']['preserved_files_unchanged'] = True
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    return summary, steps, recovery


if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        summary,_,_ = refresh_coordination()
    print(summary.to_string(index=False))
