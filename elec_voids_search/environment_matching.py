"""Optional framework-shell matching, independent of the frozen search pipeline.

Compare shell composition, bond lengths, bond angles and species-constrained
orthogonal alignment. Existing mobile ions do not define the framework shell.
"""
from dataclasses import asdict, dataclass
from itertools import permutations, product
import math

import numpy as np
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .occupied_sites import get_nonequivalent_occupied_sites


@dataclass(frozen=True)
class EnvironmentMatchConfig:
    neighbor_cutoff_A: float = 6.0
    shell_distance_ratio: float = 1.25
    radial_rms_tolerance_A: float = 0.15
    angle_rms_tolerance_deg: float = 12.0
    aligned_rms_tolerance_A: float = 0.25
    maximum_exact_permutations: int = 40320
    allow_reflection: bool = True
    symmetry_symprec_A: float = 0.10
    numeric_tolerance: float = 1e-8

    def __post_init__(self):
        for name in ['neighbor_cutoff_A','radial_rms_tolerance_A','angle_rms_tolerance_deg',
                     'aligned_rms_tolerance_A','symmetry_symprec_A']:
            if not math.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.shell_distance_ratio) or self.shell_distance_ratio <= 1:
            raise ValueError('shell_distance_ratio must be > 1')
        if not math.isfinite(self.numeric_tolerance) or self.numeric_tolerance < 0:
            raise ValueError('numeric_tolerance must be finite and nonnegative')
        if (not isinstance(self.maximum_exact_permutations,int)
                or isinstance(self.maximum_exact_permutations,bool) or self.maximum_exact_permutations < 1):
            raise ValueError('maximum_exact_permutations must be a positive integer')
        if not isinstance(self.allow_reflection,bool):
            raise ValueError('allow_reflection must be bool')

    def to_dict(self):
        return asdict(self)


def describe_framework_shell(structure, frac_point, oxidation_states, probe_element='Li',
                             probe_oxidation_state=1, config=None):
    settings = config or EnvironmentMatchConfig()
    center = structure.lattice.get_cartesian_coords(frac_point)
    neighbors = []
    for neighbor in structure.get_sites_in_sphere(center, settings.neighbor_cutoff_A):
        symbol = neighbor.specie.symbol
        if symbol == probe_element or oxidation_states[symbol] * probe_oxidation_state >= 0:
            continue
        if neighbor.nn_distance <= settings.numeric_tolerance:
            raise ValueError('Candidate coincides with a coordinating framework atom')
        neighbors.append((float(neighbor.nn_distance),symbol,np.asarray(neighbor.coords)-center))
    if not neighbors:
        raise ValueError('No oppositely charged framework neighbors inside cutoff')
    nearest = min(item[0] for item in neighbors)
    boundary = settings.shell_distance_ratio * nearest
    if boundary >= settings.neighbor_cutoff_A:
        raise ValueError('Increase neighbor_cutoff_A to cover the entire shell')
    selected = sorted((item for item in neighbors if item[0] <= boundary + settings.numeric_tolerance),
                      key=lambda item:(item[1],item[0],*item[2]))
    labels = [item[1] for item in selected]
    vectors = np.asarray([item[2] for item in selected])
    lengths = np.linalg.norm(vectors,axis=1)
    units = vectors / lengths[:,None]
    angles = {}
    for i in range(len(selected)):
        for j in range(i+1,len(selected)):
            key = tuple(sorted((labels[i],labels[j])))
            angles.setdefault(key,[]).append(float(np.degrees(np.arccos(np.clip(units[i]@units[j],-1.,1.)))))
    return {'labels':labels, 'vectors':vectors, 'lengths':lengths,
            'angles':{key:np.sort(values) for key,values in angles.items()},
            'coordination':len(selected), 'shell_radius_A':boundary,
            'composition':{symbol:labels.count(symbol) for symbol in sorted(set(labels))}}


def compare_shell_geometry(candidate, reference, config=None):
    """Require the same shell composition and separate length/angle limits."""
    settings = config or EnvironmentMatchConfig()
    result = {'similar':False,'comparison_status':'rejected','reason':'shell_composition',
              'radial_rms_A':None,'angle_rms_deg':None,'aligned_rms_A':None,'normalized_error':None}
    if candidate['composition'] != reference['composition']:
        return result
    labels = candidate['labels']
    squared = []
    for symbol in sorted(set(labels)):
        left = np.sort(candidate['lengths'][np.asarray(labels)==symbol])
        right = np.sort(reference['lengths'][np.asarray(reference['labels'])==symbol])
        squared.extend((left-right)**2)
    radial = float(np.sqrt(np.mean(squared)))
    angle_sq = [delta**2 for key,values in candidate['angles'].items()
                for delta in values-reference['angles'][key]]
    angular = float(np.sqrt(np.mean(angle_sq))) if angle_sq else 0.
    result.update(radial_rms_A=radial,angle_rms_deg=angular,reason='bond_lengths_or_angles')
    ratios = [radial/settings.radial_rms_tolerance_A,angular/settings.angle_rms_tolerance_deg]
    result['normalized_error'] = max(ratios)
    if max(ratios) > 1. + settings.numeric_tolerance:
        return result
    result.update(similar=True,comparison_status='passed',reason='similar_shell_geometry')
    return result


def compare_framework_shells(candidate, reference, config=None):
    """Origin is the site center; align without fitting translations or scale."""
    settings = config or EnvironmentMatchConfig()
    result = compare_shell_geometry(candidate,reference,settings)
    if not result['similar']:
        return result
    result.update(similar=False,comparison_status='rejected',reason='bond_lengths_or_angles')
    labels = candidate['labels']
    ratios = [result['radial_rms_A']/settings.radial_rms_tolerance_A,
              result['angle_rms_deg']/settings.angle_rms_tolerance_deg]
    # Exact label-constrained permutations; report an explicit unresolved pair
    # when the configured computation budget is exceeded, never call it similar.
    groups = [[i for i,symbol in enumerate(reference['labels']) if symbol==element]
              for element in sorted(set(labels))]
    total = math.prod(math.factorial(len(group)) for group in groups)
    if total > settings.maximum_exact_permutations:
        result.update(comparison_status='indeterminate',reason='permutation_budget')
        return result
    orders = np.asarray([sum((list(part) for part in combination),[])
                         for combination in product(*(permutations(group) for group in groups))],dtype=int)
    left = candidate['vectors']
    right = reference['vectors'][orders]
    covariance = np.einsum('ni,pnj->pij',left,right)
    u,_,vt = np.linalg.svd(covariance)
    rotations = u @ vt
    if not settings.allow_reflection:
        correction = np.ones((len(rotations),3))
        correction[:,-1] = np.linalg.det(rotations)
        rotations = (u * correction[:,None,:]) @ vt
    transformed = np.einsum('ni,pij->pnj',left,rotations)
    aligned = float(np.sqrt(np.min(np.mean(np.sum((transformed-right)**2,axis=2),axis=1))))
    ratios.append(aligned/settings.aligned_rms_tolerance_A)
    passed = max(ratios) <= 1. + settings.numeric_tolerance
    result.update(similar=passed,reason='similar_shell' if passed else 'aligned_coordinates',
                  aligned_rms_A=aligned,normalized_error=max(ratios))
    result['comparison_status'] = 'passed' if passed else 'rejected'
    return result


class FrameworkEnvironmentMatcher:
    def __init__(self, structure, oxidation_states, probe_element='Li', probe_oxidation_state=1,
                 config=None, reference_indices=None):
        if not structure.is_ordered:
            raise ValueError('Resolve partial occupancies before environment comparison')
        self.structure = structure
        self.oxidation_states = dict(oxidation_states)
        self.probe_element = probe_element
        self.probe_oxidation_state = probe_oxidation_state
        self.config = config or EnvironmentMatchConfig()
        missing = {site.specie.symbol for site in structure} - set(oxidation_states)
        if missing:
            raise ValueError(f'Missing manual oxidation states: {sorted(missing)}')
        if reference_indices is None:
            analyzer = SpacegroupAnalyzer(structure,symprec=self.config.symmetry_symprec_A)
            groups = get_nonequivalent_occupied_sites(structure,analyzer,probe_element)
            if groups.empty:
                raise ValueError('No occupied reference sites')
            reference_indices = groups.atom_indices.map(lambda group:group[0]).tolist()
        self.references = {int(index):self.describe(structure[int(index)].frac_coords)
                           for index in reference_indices}
        if not self.references:
            raise ValueError('No occupied reference sites')

    def describe(self, point):
        return describe_framework_shell(self.structure,point,self.oxidation_states,
                                       self.probe_element,self.probe_oxidation_state,self.config)

    def screen(self, candidates):
        rows = []
        for _,candidate in candidates.iterrows():
            shell = self.describe(candidate[['frac_x','frac_y','frac_z']].to_numpy(dtype=float))
            pairs = [(index,compare_framework_shells(shell,reference,self.config))
                     for index,reference in self.references.items()]
            passing = [pair for pair in pairs if pair[1]['similar']]
            index,best = min(passing or pairs,key=lambda pair:(
                pair[1]['normalized_error'] is None,
                pair[1]['normalized_error'] if pair[1]['normalized_error'] is not None else np.inf))
            row = candidate.to_dict()
            unresolved = any(pair[1]['comparison_status']=='indeterminate' for pair in pairs)
            row.update(environment_similar=bool(passing),
                       environment_status='passed' if passing else ('indeterminate' if unresolved else 'rejected'),
                       environment_reference_site_index=index,
                       environment_similar_site_indices=[pair[0] for pair in passing],
                       environment_reason=best['reason'],
                       environment_coordination=shell['coordination'],
                       environment_shell_composition=str(shell['composition']),
                       environment_radial_rms_A=best['radial_rms_A'],
                       environment_angle_rms_deg=best['angle_rms_deg'],
                       environment_aligned_rms_A=best['aligned_rms_A'])
            rows.append(row)
        extra=['environment_similar','environment_status','environment_reference_site_index',
               'environment_similar_site_indices','environment_reason','environment_coordination',
               'environment_shell_composition','environment_radial_rms_A',
               'environment_angle_rms_deg','environment_aligned_rms_A']
        return pd.DataFrame(rows,columns=list(candidates.columns)+extra)


def screen_environment_similarity(structure, candidates, oxidation_states, probe_element='Li',
                                  probe_oxidation_state=1, config=None):
    """Final optional stage; caller supplies candidates from previous screening.

    No prior stage is recomputed. Candidate coordinates and IDs are preserved.
    """
    matcher = FrameworkEnvironmentMatcher(structure,oxidation_states,probe_element,
                                          probe_oxidation_state,config=config)
    evaluated = matcher.screen(candidates)
    return {
        'environment_evaluated': evaluated,
        'environment_passed': evaluated.loc[evaluated.environment_status.eq('passed')].copy(),
        'environment_rejected': evaluated.loc[evaluated.environment_status.eq('rejected')].copy(),
        'environment_indeterminate': evaluated.loc[evaluated.environment_status.eq('indeterminate')].copy(),
        'settings': matcher.config.to_dict(),
    }
