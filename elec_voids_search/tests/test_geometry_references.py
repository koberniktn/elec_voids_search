"""Calibration contracts; run only at the user's explicit request."""
from unittest import TestCase

from elec_voids_search.config import SearchConfig
from elec_voids_search.geometry.geometry_references import OccupiedGeometryCalibration, vacancy_settings


class GeometryReferenceTests(TestCase):
    def setUp(self):
        self.calibration = OccupiedGeometryCalibration.__new__(OccupiedGeometryCalibration)
        self.calibration.settings = SearchConfig(geometry_reference_mode='occupied_sites')
        self.key = (6.,(('S',4),))
        self.calibration.types = {self.key:[{'site_index':0,'reference_ratio':.75}]}

    def test_occupied_deficit_does_not_receive_another_ten_percent(self):
        threshold,reference = self.calibration._threshold(self.key)
        self.assertEqual(threshold,.75)
        self.assertEqual(reference['site_index'],0)

    def test_unmatched_environment_keeps_percentage_threshold(self):
        threshold,reference = self.calibration._threshold((6.,(('S',6),)))
        self.assertEqual(threshold,.9)
        self.assertIsNone(reference)

    def test_roomy_reference_cannot_tighten_percentage_criterion(self):
        self.calibration.types[self.key][0]['reference_ratio'] = 1.1
        self.assertEqual(self.calibration._threshold(self.key)[0],.9)

    def test_atom_overrides_follow_atoms_when_central_site_is_deleted(self):
        original = SearchConfig(geometry_host_radius_overrides_A={'0':.5,'2':.6,'3':.7,'S':1.84})
        shifted = vacancy_settings(original,2)
        self.assertEqual(shifted.geometry_host_radius_overrides_A,{'0':.5,'2':.7,'S':1.84})
        self.assertEqual(original.geometry_host_radius_overrides_A['2'],.6)
