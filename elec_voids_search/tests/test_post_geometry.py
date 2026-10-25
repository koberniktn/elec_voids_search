"""Prepared tests; execute only on an explicit user request."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

from elec_voids_search.descriptors.framework_voronoi import framework_voronoi_neighbors, FrameworkVoronoiError
from elec_voids_search.filtering.voronoi_coordination_config import VoronoiCoordinationConfig
from elec_voids_search.filtering.weighted_voronoi import (describe_weighted_voronoi,compare_weighted_voronoi,
                                               WeightedVoronoiMatcher)
from elec_voids_search.filtering.post_geometry import screen_geometry_candidates
from elec_voids_search.filtering.post_geometry_config import PostGeometryConfig
from elec_voids_search.tools.saved_screening import continue_from_saved_geometry


class FrameworkVoronoiTests(TestCase):
    def setUp(self):
        self.host = Structure(Lattice.cubic(6),['Li','O','O','O'],
                              [[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5]])
        self.charges = {'Li':1,'O':-2}

    def test_exact_cube_geometry_and_descriptor_identity(self):
        descriptor = describe_weighted_voronoi(self.host,[0,0,0],self.charges)
        self.assertAlmostEqual(descriptor['scalars']['voronoi_volume_A3'],27.)
        self.assertAlmostEqual(descriptor['scalars']['cn_effective'],6.)
        self.assertAlmostEqual(descriptor['scalars']['distance_mean_A'],3.)
        self.assertTrue(compare_weighted_voronoi(descriptor,descriptor)['similar'])

    def test_mobile_occupancy_does_not_change_framework_descriptor(self):
        occupied = self.host.copy();occupied.append('Li',[.2,.2,.2])
        a = describe_weighted_voronoi(self.host,[0,0,0],self.charges)
        b = describe_weighted_voronoi(occupied,[0,0,0],self.charges)
        score = compare_weighted_voronoi(a,b)
        self.assertTrue(score['similar'])
        self.assertAlmostEqual(score['combined_score'],0.)

    def test_probe_images_do_not_clip_octahedral_cell(self):
        host = Structure(Lattice.cubic(6),['Li','O'],[[0,0,0],[.5,.5,.5]])
        descriptor = describe_weighted_voronoi(host,[0,0,0],self.charges)
        # Eight (±3,±3,±3) O images bound |x|+|y|+|z|<=4.5.
        # A periodically repeated artificial query would cut the x=4.5 vertex.
        self.assertAlmostEqual(descriptor['scalars']['voronoi_volume_A3'],121.5)
        self.assertAlmostEqual(descriptor['scalars']['maximum_vertex_radius_A'],4.5)
        self.assertGreater(descriptor['scalars']['cutoff_A'],9.)

    def test_uncertified_cell_is_not_accepted(self):
        host = Structure(Lattice.cubic(6),['Li','O'],[[0,0,0],[.5,.5,.5]])
        with self.assertRaises(FrameworkVoronoiError):
            framework_voronoi_neighbors(host,[0,0,0],config=VoronoiCoordinationConfig(maximum_cutoff_A=6.))

    def test_periodic_and_supercell_representation(self):
        a = describe_weighted_voronoi(self.host,[0,0,0],self.charges)
        b = describe_weighted_voronoi(self.host,[1,-1,2],self.charges)
        expanded = self.host.copy();expanded.make_supercell([2,1,1])
        c = describe_weighted_voronoi(expanded,[0,0,0],self.charges)
        self.assertAlmostEqual(compare_weighted_voronoi(a,b)['combined_score'],0.,places=10)
        self.assertAlmostEqual(compare_weighted_voronoi(a,c)['combined_score'],0.,places=10)

    def test_distorted_and_chemically_different_environments(self):
        a = describe_weighted_voronoi(self.host,[0,0,0],self.charges)
        larger = Structure(Lattice.cubic(10),['Li','O','O','O'],
                           [[0,0,0],[.5,0,0],[0,.5,0],[0,0,.5]])
        b = describe_weighted_voronoi(larger,[0,0,0],self.charges)
        self.assertFalse(compare_weighted_voronoi(a,b)['similar'])
        chloride = Structure(self.host.lattice,['Li','Cl','Cl','Cl'],self.host.frac_coords)
        c = describe_weighted_voronoi(chloride,[0,0,0],{'Li':1,'Cl':-1})
        self.assertFalse(compare_weighted_voronoi(a,c)['similar'])

    def test_element_pair_angles_distinguish_cis_and_trans(self):
        vectors = np.asarray([[2,0,0],[-2,0,0],[0,2,0],[0,-2,0],[0,0,2],[0,0,-2]],dtype=float)
        def environment(chlorine_indices):
            species = ['Li']+['Cl' if i in chlorine_indices else 'O' for i in range(6)]
            return Structure(Lattice.cubic(12),species,
                             np.vstack([np.zeros(3),vectors/12])%1)
        charges = {'Li':1,'O':-2,'Cl':-1}
        cis = describe_weighted_voronoi(environment({0,2}),[0,0,0],charges)
        trans = describe_weighted_voronoi(environment({0,1}),[0,0,0],charges)
        self.assertAlmostEqual(cis['scalars']['voronoi_volume_A3'],
                               trans['scalars']['voronoi_volume_A3'])
        self.assertAlmostEqual(cis['primary']['angle_profiles'][('Cl','Cl')][0][0],90.)
        self.assertAlmostEqual(trans['primary']['angle_profiles'][('Cl','Cl')][0][0],180.)
        comparison = compare_weighted_voronoi(cis,trans)
        self.assertGreater(comparison['primary_angular_difference_deg'],0.)
        self.assertFalse(comparison['similar'])
        self.assertAlmostEqual(comparison['primary_angular_difference_deg'],
                               compare_weighted_voronoi(trans,cis)['primary_angular_difference_deg'])

    def test_tolerances_can_change_without_rebuilding_descriptors(self):
        matcher = WeightedVoronoiMatcher(self.host,self.charges)
        candidates = pd.DataFrame({'candidate_id':[1],'frac_x':[0.],'frac_y':[0.],'frac_z':[0.]})
        matcher.screen(candidates)
        size = len(matcher.cache)
        matcher.screen(candidates,replace(matcher.config,distance_tolerance_A=.3))
        self.assertEqual(len(matcher.cache),size)
        with self.assertRaises(ValueError):
            matcher.screen(candidates,replace(matcher.config,initial_cutoff_A=8.))
        with self.assertRaises(ValueError):
            matcher.screen(candidates,replace(matcher.config,first_shell_distance_ratio=1.3))

    def test_volume_cannot_compensate_first_shell_composition(self):
        reference = describe_weighted_voronoi(self.host,[0,0,0],self.charges)
        candidate = deepcopy(reference)
        candidate['first_shell']['composition'] = {'O':5}
        comparison = compare_weighted_voronoi(candidate,reference)
        self.assertEqual(comparison['combined_score'],0.)
        self.assertFalse(comparison['similar'])
        self.assertIn('first_shell_shell_composition',comparison['reason'])

    def test_first_shell_lengths_and_angles_are_independent_limits(self):
        reference = describe_weighted_voronoi(self.host,[0,0,0],self.charges)
        for metric in ['lengths','angles']:
            candidate = deepcopy(reference)
            if metric=='lengths':
                candidate['first_shell']['lengths'] += .16
            else:
                for angles in candidate['first_shell']['angles'].values():
                    angles -= 13.
            comparison = compare_weighted_voronoi(candidate,reference)
            self.assertEqual(comparison['combined_score'],0.)
            self.assertFalse(comparison['similar'])
            self.assertIn('first_shell_bond_lengths_or_angles',comparison['reason'])


class BranchRoutingTests(TestCase):
    def setUp(self):
        self.host = Structure(Lattice.cubic(6),['Li','O'],[[0,0,0],[.5,.5,.5]])
        self.candidates = pd.DataFrame({'candidate_id':[1,2,3,4],
            'frac_x':[.1,.2,.3,.4],'frac_y':[.2]*4,'frac_z':[.3]*4,
            'geometry_status':['passed']*4})
        self.calls = {'coordination':[],'li_distance':[]}
        calls = self.calls
        class FakeMatcher:
            reference_table = pd.DataFrame({'site_index':[0]})
            def __init__(self,*args,**kwargs):pass
            def screen(self,table):
                calls['coordination'].extend(table.candidate_id.tolist())
                output = table.copy()
                output['coordination_status'] = np.where(output.candidate_id.isin([1,2]),'passed','rejected')
                return output
        self.matcher = FakeMatcher
        def separation(structure,table,config):
            calls['li_distance'].extend(table.candidate_id.tolist())
            output = table.copy()
            output['probe_distance_status'] = np.where(output.candidate_id.isin([2,3]),'passed','rejected')
            return dict(probe_distance_evaluated=output,occupied_probe_sites=pd.DataFrame(),
                        occupied_probe_nonequivalent=pd.DataFrame())
        self.separation = separation

    def run_paths(self,paths):
        with patch('elec_voids_search.filtering.post_geometry.WeightedVoronoiMatcher',self.matcher), \
             patch('elec_voids_search.filtering.post_geometry.screen_probe_separation',self.separation):
            return screen_geometry_candidates(self.host,self.candidates,{'Li':1,'O':-2},
                                               PostGeometryConfig(paths=paths))

    def test_independent_branches_and_sequential_intersections(self):
        result = self.run_paths({'coordination':('coordination',),'li':('li_distance',),
            'coord_then_li':('coordination','li_distance'),
            'li_then_coord':('li_distance','coordination')})
        expected = {'coordination':[1,2],'li':[2,3],'coord_then_li':[2],'li_then_coord':[2]}
        for name,ids in expected.items():
            self.assertEqual(result['branches'][name]['passed'].candidate_id.tolist(),ids)
        self.assertEqual(self.calls['coordination'],[1,2,3,4])
        self.assertEqual(self.calls['li_distance'],[1,2,3,4])
        self.assertEqual(result['candidates'].filter_comparison_group.tolist(),
                         ['coordination_only','passed_both','li_distance_only','rejected_both'])

    def test_skipped_filter_is_not_reported_as_rejection(self):
        result = self.run_paths({'sequence':('coordination','li_distance')})
        self.assertEqual(self.calls['li_distance'],[1,2])
        status = result['candidates'].set_index('candidate_id').probe_distance_status
        self.assertEqual(status.loc[3],'not_evaluated')
        self.assertEqual(status.loc[4],'not_evaluated')
        self.assertEqual(len(result['branches']['sequence']['rejected']),3)

    def test_empty_input_does_not_construct_matchers(self):
        self.candidates = self.candidates.iloc[:0]
        result = self.run_paths({'a':('coordination',),'b':('li_distance',)})
        self.assertEqual(self.calls,{'coordination':[],'li_distance':[]})
        self.assertEqual(len(result['candidates']),0)
        self.assertTrue((result['summary'].n_input==0).all())

    def test_saved_geometry_continuation_ignores_old_li_rejections(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder/'input.cif';source.write_text('fixture input; Structure reader is mocked')
            saved = self.candidates.copy();saved['in_geometry'] = [True,True,False,False]
            saved['probe_distance_status'] = 'rejected'
            saved.to_csv(folder/'candidates.csv',index=False)
            with patch('elec_voids_search.tools.saved_screening.Structure.from_file',return_value=self.host), \
                 patch('elec_voids_search.screening_pipeline.find_candidate_sites',side_effect=AssertionError('No rerun')), \
                 patch('elec_voids_search.tools.saved_screening.screen_geometry_candidates',return_value={}) as function:
                result = continue_from_saved_geometry(source,folder/'candidates.csv',{'Li':1,'O':-2})
            self.assertEqual(function.call_args.args[1].candidate_id.tolist(),[1,2])
            self.assertFalse(result['provenance']['previous_stages_rerun'])
