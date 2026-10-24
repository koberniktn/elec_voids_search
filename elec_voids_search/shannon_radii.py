"""Степени окисления, координация и ионные радиусы средствами pymatgen."""

from functools import lru_cache
import numpy as np
import pandas as pd
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Species


def oxidized_structure(structure):
    """Возвращаем копию со степенями окисления на каждом сайте."""
    if not len(structure) or not structure.is_ordered:
        raise ValueError("Нужна непустая структура без частичной заселённости")
    if all(hasattr(site.specie, "oxi_state") for site in structure):
        return structure.copy()
    # Заряды определяет встроенный анализ валентностей связей.
    return BVAnalyzer().get_oxi_state_decorated_structure(structure.copy())


def crystalnn_finder(cutoff=12.0):
    """КЧ определяем без радиусов: они будут найдены только после этого."""
    return CrystalNN(
        cation_anion=True,
        distance_cutoffs=None,
        x_diff_weight=0,
        porous_adjustment=False,
        search_cutoff=cutoff,
    )


@lru_cache(maxsize=None)
def shannon_radius(element, charge, coordination, spin=""):
    """Ионный радиус Шеннона в Å; одинаковые запросы кэшируем."""
    roman = ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX",
             "X", "XI", "XII")
    if not 1 <= coordination <= 12:
        raise ValueError(f"Нет обозначения для КЧ {coordination}")
    try:
        return float(Species(element, charge).get_shannon_radius(
            roman[coordination], spin=spin, radius_type="ionic"
        ))
    except KeyError:
        raise ValueError(f"Нет радиуса Шеннона: {element}, заряд {charge}, "
                         f"КЧ {coordination}, спин '{spin}'") from None


def prepare_shannon_radii(structure, mobile="Li", cutoff=12.0, spin=""):
    """Радиусы исходных сайтов. Результат совместим с site_descriptors().

    Степени окисления — BVAnalyzer, КЧ — CrystalNN,
    радиусы — Species.get_shannon_radius. Исходную структуру не изменяем.
    Если радиуса для найденного КЧ нет, сообщаем об этом без подмены.
    """
    charged = oxidized_structure(structure)
    finder = crystalnn_finder(cutoff)
    rows = []
    for index, site in enumerate(charged):
        charge = site.specie.oxi_state
        cn = int(finder.get_cn(charged, index, use_weights=False))
        try:
            radius = shannon_radius(site.specie.symbol, charge, cn, spin)
        except ValueError as error:
            raise ValueError(f"Сайт {index}: {error}") from error
        rows.append({
            "site_index": index,
            "element": site.specie.symbol,
            "oxidation_state": charge,
            "valence": abs(charge),  # Валентность в используемой ионной модели.
            "coordination": cn,
            "radius_A": radius,
        })
    table = pd.DataFrame(rows)
    radii = table["radius_A"].to_numpy()
    return {
        "table": table, "site_table": table,
        "site_radii": radii,
        "framework_radii": np.where(table["element"] != mobile, radii, np.nan),
        "site_charges": table["oxidation_state"].to_numpy(),
        "mobile": mobile,
        "species": [str(site.specie) for site in structure],
        "frac_coords": structure.frac_coords.copy(),
        "lattice": structure.lattice.matrix.copy(),
        "radius_type": "Shannon effective ionic radius",
        "coordination_method": "CrystalNN (without radius cutoffs)",
    }
