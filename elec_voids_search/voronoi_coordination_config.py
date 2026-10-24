"""Visible working defaults for weighted framework-Voronoi comparison."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class VoronoiCoordinationConfig:
    initial_cutoff_A: float = 6.0
    maximum_cutoff_A: float = 24.0
    cutoff_growth: float = 1.5
    minimum_face_area_A2: float = 1e-12
    solid_angle_sum_tolerance: float = 1e-5
    significant_face_fraction: float = 0.01  # report only; no truncation of the cell
    numeric_tolerance: float = 1e-8
    symmetry_symprec_A: float = 0.10
    symmetry_angle_tolerance_deg: float = 5.0

    chemical_fraction_tolerance: float = 0.15  # total variation of element fractions
    effective_cn_tolerance: float = 1.0
    distance_tolerance_A: float = 0.20
    size_length_tolerance_A: float = 0.20  # difference of V**(1/3)
    angular_tolerance_deg: float = 12.0
    sphericity_tolerance: float = 0.10
    primary_environment_weight: float = 0.75  # opposite-charge neighbors
    first_shell_distance_ratio: float = 1.25  # d <= ratio * nearest opposite-charge distance
    first_shell_radial_tolerance_A: float = 0.15
    first_shell_angular_tolerance_deg: float = 12.0

    coordination_block_weight: float = 0.25
    distance_block_weight: float = 0.35
    size_block_weight: float = 0.15
    shape_block_weight: float = 0.25
    combined_score_limit: float = 1.0
    maximum_block_score: float = 2.0
    maximum_primary_score: float = 1.0
    reference_error_policy: str = 'indeterminate'  # or raise

    def __post_init__(self):
        positive = ['initial_cutoff_A','maximum_cutoff_A','solid_angle_sum_tolerance',
                    'symmetry_symprec_A','chemical_fraction_tolerance','effective_cn_tolerance',
                    'distance_tolerance_A','size_length_tolerance_A','angular_tolerance_deg',
                    'sphericity_tolerance','combined_score_limit','maximum_block_score','maximum_primary_score',
                    'first_shell_radial_tolerance_A','first_shell_angular_tolerance_deg']
        for name in positive:
            value = getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.maximum_cutoff_A < self.initial_cutoff_A:
            raise ValueError('maximum_cutoff_A must be >= initial_cutoff_A')
        if not math.isfinite(self.first_shell_distance_ratio) or self.first_shell_distance_ratio <= 1:
            raise ValueError('first_shell_distance_ratio must be finite and > 1')
        if not math.isfinite(self.cutoff_growth) or self.cutoff_growth <= 1:
            raise ValueError('cutoff_growth must be > 1')
        for name in ['minimum_face_area_A2','numeric_tolerance']:
            value = getattr(self,name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ['significant_face_fraction','primary_environment_weight']:
            value = getattr(self,name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        weights = self.block_weights
        if not all(math.isfinite(value) and value >= 0 for value in weights) or sum(weights) <= 0:
            raise ValueError('Block weights must be nonnegative with positive sum')
        angle = self.symmetry_angle_tolerance_deg
        if not math.isfinite(angle) or not (angle == -1 or 0 < angle <= 180):
            raise ValueError('Invalid symmetry_angle_tolerance_deg')
        if self.reference_error_policy not in {'raise','indeterminate'}:
            raise ValueError('reference_error_policy must be raise or indeterminate')

    @property
    def block_weights(self):
        return (self.coordination_block_weight,self.distance_block_weight,
                self.size_block_weight,self.shape_block_weight)

    def to_dict(self):
        return asdict(self)
