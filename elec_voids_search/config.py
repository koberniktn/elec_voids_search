"""Настройки поиска и геометрии; основной режим - пропускаем кандидатов со свободным радиусом, 
больше или равном радиусу Шенона для пробного атома с фиксированным процентнымм допуском.
Дополнительно реализована возможность вместо процентного допуска анализировать самый большой
свободный радиус, требующийся для уже занятых сайтовв в структуре - пока тестовый вариант"""

from dataclasses import asdict, dataclass, field, fields, replace
import math

from pymatgen.core import Element


@dataclass(frozen=True)
class SearchConfig:
    """Параметры сетки, Вороного, объединения, симметрии и геометрии."""

    # Сетка: шаг и параметры поиска сырых максимумов.
    grid_resolution_A: float = 0.2
    grid_search_cutoff_A: float = 5.0
    grid_symprec_A: float | None = 0.1
    grid_minima_window: int = 3
    grid_r_min_A: float = 1.0
    grid_empty_framework: bool = False
    grid_verbose: bool = False
    probe_element: str = 'Li'
    probe_oxidation_state: float = 1

    # Вороной: периодические образы и численные допуски.
    voronoi_image_range: int = 1
    voronoi_boundary_tolerance: float = 1e-09
    voronoi_qhull_options: str | None = None

    # Общая кластеризация кандидатов двух методов.
    merge_distance_A: float | None = 0.75 #сливаем кандидатов, находящихся ближе этого предела друг к другу, в один кластер
    merge_representative: str = 'largest_center_distance'
    merge_grouping: str = 'connected'

    # Симметрийный отсев выполняется до геометрии.
    symmetry_symprec_A: float = 0.1
    symmetry_tolerance_A: float = 0.1
    symmetry_angle_tolerance_deg: float = 5.0

    # Основная геометрия — полный радиус пробника с процентным допуском.
    geometry_minimum_fit_fraction: float = 1.0
    geometry_allow_ionic_fallback: bool = True
    geometry_probe_radius_override_A: float | None = None
    export_cif: bool = False
    geometry_probe_spin: str = ''
    geometry_overlap_tolerance_percent: float = 10.0 #процентный допуск, выбран импирически на основе тестовых соединений

    # Калибровка по занятым сайтам включается отдельно.
    geometry_reference_mode: str = 'percentage'
    geometry_reference_shell_distance_ratio: float = 1.25
    geometry_reference_margin_percent: float = 0.0
    geometry_numeric_tolerance_A: float = 1e-07
    geometry_neighbor_cutoff_A: float = 12.0

    # Заряды задаём вручную; радиусы можно переопределить явно.
    geometry_oxidation_states: dict = field(default_factory=dict)
    geometry_host_radius_overrides_A: dict = field(default_factory=dict)
    geometry_spin: str = ''


    def __post_init__(self):
        for name in (
            'grid_resolution_A',
            'grid_search_cutoff_A',
            'symmetry_symprec_A',
            'symmetry_tolerance_A',
            'geometry_neighbor_cutoff_A',
            'geometry_numeric_tolerance_A',
        ):
            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')

        for name in ('grid_symprec_A', 'merge_distance_A', 'geometry_probe_radius_override_A'):
            value = getattr(self, name)

            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be None or finite and positive')

        if not math.isfinite(self.grid_r_min_A) or self.grid_r_min_A < 0:
            raise ValueError('grid_r_min_A must be finite and nonnegative')

        if (
            (not isinstance(self.grid_minima_window, int))
            or isinstance(self.grid_minima_window, bool)
            or (self.grid_minima_window < 3)
            or (self.grid_minima_window % 2 != 1)
        ):
            raise ValueError('grid_minima_window must be an odd integer >= 3')

        if (
            (not isinstance(self.voronoi_image_range, int))
            or isinstance(self.voronoi_image_range, bool)
            or (self.voronoi_image_range < 1)
        ):
            raise ValueError('voronoi_image_range must be an integer >= 1')

        if not isinstance(self.probe_element, str):
            raise ValueError('probe_element must be a chemical element symbol')
        Element(self.probe_element)

        if (
            isinstance(self.probe_oxidation_state, bool)
            or (not isinstance(self.probe_oxidation_state, (int, float)))
            or (not math.isfinite(self.probe_oxidation_state))
            or (self.probe_oxidation_state == 0)
            or (not float(self.probe_oxidation_state).is_integer())
        ):
            raise ValueError('probe_oxidation_state must be a finite nonzero integer')

        if not 0 < self.geometry_minimum_fit_fraction <= 1:
            raise ValueError('geometry_minimum_fit_fraction must be in (0, 1]')

        if (
            (not math.isfinite(self.voronoi_boundary_tolerance))
            or (not 0 <= self.voronoi_boundary_tolerance < 0.5)
        ):
            raise ValueError('voronoi_boundary_tolerance must be finite and in [0, 0.5)')

        if self.voronoi_qhull_options is not None and (not isinstance(self.voronoi_qhull_options, str)):
            raise ValueError('voronoi_qhull_options must be None or a string')

        if self.merge_grouping not in {'representative', 'connected'}:
            raise ValueError('merge_grouping must be representative or connected')

        if self.merge_representative not in {'first', 'grid', 'voronoi', 'largest_center_distance'}:
            raise ValueError('Unknown merge_representative policy')

        if (
            (not math.isfinite(self.symmetry_angle_tolerance_deg))
            or (not (self.symmetry_angle_tolerance_deg == -1 or 0 < self.symmetry_angle_tolerance_deg <= 180))
        ):
            raise ValueError('symmetry_angle_tolerance_deg must be -1 or in (0, 180]')

        if (
            (not math.isfinite(self.geometry_overlap_tolerance_percent))
            or (not 0 <= self.geometry_overlap_tolerance_percent < 100)
        ):
            raise ValueError('geometry_overlap_tolerance_percent must be finite and in [0, 100)')

        if self.geometry_reference_mode not in {'percentage', 'occupied_sites'}:
            raise ValueError('geometry_reference_mode must be percentage or occupied_sites')

        if (
            (not math.isfinite(self.geometry_reference_shell_distance_ratio))
            or (self.geometry_reference_shell_distance_ratio <= 1)
        ):
            raise ValueError('geometry_reference_shell_distance_ratio must be finite and > 1')

        if (
            (not math.isfinite(self.geometry_reference_margin_percent))
            or (not 0 <= self.geometry_reference_margin_percent < 100)
        ):
            raise ValueError('geometry_reference_margin_percent must be finite and in [0, 100)')

        for name in ('geometry_oxidation_states', 'geometry_host_radius_overrides_A'):
            values = getattr(self, name)

            if not isinstance(values, dict):
                raise ValueError(f'{name} must be a dictionary')

            for key, value in values.items():
                if (
                    (not isinstance(key, str))
                    or (not isinstance(value, (int, float)))
                    or (not math.isfinite(value))
                ):
                    raise ValueError(f'Invalid entry in {name}: {key}')

                if name == 'geometry_oxidation_states' and (not float(value).is_integer()):
                    raise ValueError('Oxidation states must be integers for ionic-radius lookup')

                if name == 'geometry_host_radius_overrides_A' and value <= 0:
                    raise ValueError('Host radius overrides must be positive')

        if (
            (self.geometry_spin not in {'', 'High Spin', 'Low Spin'})
            or (self.geometry_probe_spin not in {'', 'High Spin', 'Low Spin'})
        ):
            raise ValueError('geometry_spin must be empty, High Spin or Low Spin')

        for name in ('grid_empty_framework', 'grid_verbose', 'geometry_allow_ionic_fallback', 'export_cif'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be bool')


    @property
    def probe_atomic_number(self):
        """Атомный номер пробного элемента для сеточного поиска."""

        return Element(self.probe_element).Z


    def to_dict(self):
        return asdict(self)


    def geometry_allowed_overlap_A(self, probe_radius_A):
        """Переводим допуск из процентов локального радиуса пробника в Å."""

        return probe_radius_A * self.geometry_overlap_tolerance_percent / 100.0


    def grid_params(self):
        """Передаём настройки в исходный сеточный генератор."""

        return {
            'r_min': self.grid_r_min_A,
            'r_cut': self.grid_search_cutoff_A,
            'resolution': self.grid_resolution_A,
            'symprec': self.grid_symprec_A,
            'empty_framework': self.grid_empty_framework,
            'verbose': self.grid_verbose,
        }
DEFAULT_CONFIG = SearchConfig()


def resolve_config(config=None, **overrides):
    """Объединяем настройки; принимаем только имена полей SearchConfig."""

    if config is None:
        config = DEFAULT_CONFIG
    elif isinstance(config, dict):
        config = SearchConfig(**config)
    elif not isinstance(config, SearchConfig):
        raise TypeError('config must be SearchConfig, dict or None')

    # Явные переопределения используют те же имена, что и конфигурация.

    unknown = set(overrides) - {field.name for field in fields(SearchConfig)}

    if unknown:
        raise TypeError(f'Unknown search parameters: {sorted(unknown)}')

    return replace(config, **overrides)
