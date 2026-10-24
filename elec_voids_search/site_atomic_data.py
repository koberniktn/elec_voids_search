"""Координация, степень окисления и радиусы для каждого атома структуры"""

from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN


def get_cn_for_ion(index, charged, finder=None, cutoff=12.0):
    """
    Координационное число атома с индексом index в структуре charged - уже со степенями окисления
    """
    if finder is None:
        finder = CrystalNN(
                cation_anion=True,
                distance_cutoffs=None,
                x_diff_weight=0,
                porous_adjustment=False,
                search_cutoff=cutoff,
            )

    cn = int(finder.get_cn(charged, index, use_weights=False)) #координационное число

    return cn


def get_shannon_radius(specie, cn, spin=""):
    # Радиус Шеннона зависит от заряда, КЧ и спина
    roman = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI",
                     7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII"}
    
    shannon_radius = None
    if cn in roman:
        try:
            shannon_radius = float(specie.get_shannon_radius(
                roman[cn], spin=spin, radius_type="ionic"
            ))
        except KeyError:
            pass  # Такой записи в таблице Шеннона нет

    return shannon_radius


def get_site_atomic_data(structure, cutoff=12.0, spin=""):
    """
    Возвращаем {индекс атома: словарь свойств}. Все радиусы — в Å

    Отсутствующий радиус Шеннона записываем как None
    spin нужен для ионов с разными высоко- и низкоспиновыми радиусами
    """
    charged = structure.copy()
    # Если заряды уже записаны, сохраняем их. Иначе определяем через pymatgen
    if not all(hasattr(site.specie, "oxi_state") for site in charged):
        charged = BVAnalyzer().get_oxi_state_decorated_structure(charged)

    finder = CrystalNN(
        cation_anion=True,
        distance_cutoffs=None,
        x_diff_weight=0,
        porous_adjustment=False,
        search_cutoff=cutoff,
    )
    
    result = {}

    for index, site in enumerate(charged):
        specie = site.specie

        atomic_radius = specie.element.atomic_radius
        # Стандартный ионный радиус pymatgen не зависит от найденного КЧ
        ionic_radius = specie.ionic_radius

        cn = get_cn_for_ion(index, charged, finder=finder, cutoff=cutoff)
        shannon_radius = get_shannon_radius(specie, cn, spin=spin)

        result[index] = {
            "element": specie.symbol,
            "coordination": cn,
            "oxidation_state": specie.oxi_state,
            "atomic_radius_A": None if atomic_radius is None else float(atomic_radius),
            "ionic_radius_A": None if ionic_radius is None else float(ionic_radius),
            "shannon_radius_A": shannon_radius,
        }

    return charged, result
