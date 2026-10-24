"""Original Voronoi scalar descriptors, copied without changing formatting."""
import numpy as np


def neighbor_weights(neighbors, element=None):
    # Для выбранного элемента остальные соседи получают нулевой вес.
    return np.asarray([
        n["solid_angle"] if element is None or n["element"] == element else 0.0
        for n in neighbors
    ], dtype=float)


def calc_cn_raw(neighbors):
    """Число всех соседей по граням Вороного, включая периодические образы."""
    return len(neighbors)


def calc_cn_significant(neighbors, threshold=0.01):
    """Число соседей без очень маленьких граней: порог относительно наибольшего телесного угла."""
    angles = np.asarray([n["solid_angle"] for n in neighbors])
    # Порог относительно максимального угла: 0.01 означает 1%.
    return int(np.count_nonzero(angles > threshold * angles.max()))


def calc_cn_effective(neighbors):
    """Эффективное число соседей: 1 / Σp², где p — доля телесного угла. При равных весах равно их числу."""
    # Здесь веса только геометрические: p_i = Ω_i / ΣΩ_i.
    angles = np.asarray([n["solid_angle"] for n in neighbors])
    fractions = angles / angles.sum()
    return float(1 / np.sum(fractions ** 2))


def calc_element_fraction(neighbors, element):
    """Доля элемента в окружении по телесным углам; сумма долей всех элементов равна единице."""
    return float(neighbor_weights(neighbors, element).sum()
                 / neighbor_weights(neighbors).sum())


def calc_distance_min(neighbors, element=None):
    """Минимальное расстояние до центра соседа выбранного элемента, в Å."""
    distances = np.asarray([n["distance"] for n in neighbors])
    weights = neighbor_weights(neighbors, element)
    return float(distances[weights > 0].min()) if np.any(weights > 0) else np.nan


def calc_distance_mean(neighbors, element=None):
    """Среднее расстояние до центров соседей с весами по телесным углам, в Å."""
    distances = np.asarray([n["distance"] for n in neighbors])
    weights = neighbor_weights(neighbors, element)
    return float(np.average(distances, weights=weights)) if weights.sum() else np.nan


def calc_distance_std(neighbors, element=None):
    """Разброс расстояний вокруг взвешенного среднего, в Å."""
    distances = np.asarray([n["distance"] for n in neighbors])
    weights = neighbor_weights(neighbors, element)
    if not weights.sum():
        return np.nan
    mean = np.average(distances, weights=weights)
    return float(np.sqrt(np.average((distances - mean) ** 2, weights=weights)))


def calc_distance_cv(neighbors):
    """Относительный разброс расстояний: стандартное отклонение / среднее."""
    return calc_distance_std(neighbors) / calc_distance_mean(neighbors)


def calc_voronoi_volume(neighbors):
    """Объём полной ячейки Вороного вокруг точки, в Å³; это не свободный объём полости."""
    # volume в pymatgen — объём пирамиды от центра до одной грани.
    return float(sum(n["volume"] for n in neighbors))


def calc_voronoi_sphericity(neighbors):
    """Сферичность ячейки: 1 для сферы, меньше 1 для многогранника."""
    volume = calc_voronoi_volume(neighbors)
    area = sum(n["area"] for n in neighbors)
    return float(np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area)
