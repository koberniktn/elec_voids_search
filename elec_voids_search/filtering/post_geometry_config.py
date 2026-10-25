"""Настройки веток фильтрации: координация и расстояние до занятого пробника."""

from dataclasses import dataclass, field

from collections.abc import Mapping

from types import MappingProxyType

from pymatgen.core import Element

from .probe_separation import ProbeSeparationConfig
from .voronoi_coordination_config import VoronoiCoordinationConfig


@dataclass(frozen=True)
class PostGeometryConfig:
    """Пробный ион, настройки двух фильтров и явные последовательности их запуска."""
    probe_element: str = 'Li'
    probe_oxidation_state: float = 1
    coordination: VoronoiCoordinationConfig = field(default_factory=VoronoiCoordinationConfig)
    separation: ProbeSeparationConfig | None = None
    paths: dict = field(
        default_factory=lambda: {'coordination': ('coordination',), 'li_distance': ('li_distance',)},
    )
    export_cif: bool = False
    export_rejected_cif: bool = True


    def __post_init__(self):
        import math
        Element(self.probe_element)

        if (
            isinstance(self.probe_oxidation_state, bool)
            or (not math.isfinite(self.probe_oxidation_state))
            or (self.probe_oxidation_state == 0)
        ):
            raise ValueError('probe_oxidation_state must be finite and nonzero')

        if not isinstance(self.coordination, VoronoiCoordinationConfig):
            raise TypeError('coordination must be VoronoiCoordinationConfig')

        if self.separation is not None:
            if not isinstance(self.separation, ProbeSeparationConfig):
                raise TypeError('separation must be ProbeSeparationConfig or None')

            if self.separation.probe_element != self.probe_element:
                raise ValueError('Both filters must use the same probe_element')

        if not isinstance(self.paths, Mapping) or not self.paths:
            raise ValueError('paths must be a nonempty mapping of names to filter sequences')

        for name, path in self.paths.items():
            if (
                (not isinstance(name, str))
                or (not name)
                or any(
                    (c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name),
                )
            ):
                raise ValueError('Path names may contain letters, digits, underscore and hyphen')

            if not isinstance(path, (list, tuple)) or not path:
                raise ValueError('Each path must be a nonempty list or tuple')

            if any((key not in {'coordination', 'li_distance'} for key in path)):
                raise ValueError('Available filters: coordination, li_distance')

            if len(set(path)) != len(path):
                raise ValueError('Do not repeat a filter inside one path')
        object.__setattr__(
            self,
            'paths',
            MappingProxyType({name: tuple(path) for name, path in self.paths.items()}),
        )

        if not isinstance(self.export_cif, bool) or not isinstance(self.export_rejected_cif, bool):
            raise TypeError('CIF export settings must be bool')


    @property
    def resolved_separation(self):
        """Используем явную настройку расстояний или дефолт для выбранного пробника."""

        return self.separation or ProbeSeparationConfig(probe_element=self.probe_element)


    def to_dict(self):
        return dict(
            probe_element=self.probe_element,
            probe_oxidation_state=self.probe_oxidation_state,
            coordination=self.coordination.to_dict(),
            separation=self.resolved_separation.to_dict(),
            paths={name: list(path) for name, path in self.paths.items()},
            export_cif=self.export_cif,
            export_rejected_cif=self.export_rejected_cif,
        )
