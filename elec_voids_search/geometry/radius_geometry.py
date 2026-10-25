"""Geometry blocks copied verbatim from the original Shannon screening script."""
import numpy as np


def make_point_geometry(structure, site_radii):
    def pbc_distance(frac_a, frac_b):
        return float(structure.lattice.get_distance_and_image(frac_a, frac_b)[0])

    def site_hard_radius(atom_index):
        return site_radii[atom_index]

    def point_geometry(frac_point):
        free_radius = np.inf #свободный радиус полости
        nearest_center_distance = np.inf #расстояние до ближайшего атомного центра
        limiting_atom_index = -1 #атом, ограничивающий размер полости

        for atom_index, site in enumerate(structure):
            distance = pbc_distance(frac_point, site.frac_coords) #мин расстояние от вершины Воронова до центра текущего атома
            atom_clearance = distance - site_hard_radius(atom_index) #рас-ие от вершины до конца пооверхности атома (учитываем его радиус)

            nearest_center_distance = min(nearest_center_distance, distance)
            if atom_clearance < free_radius:
                free_radius = atom_clearance
                limiting_atom_index = atom_index

        return free_radius, nearest_center_distance, limiting_atom_index

    return point_geometry


def radius_selector():
    def ionic_radius(data):
        radius = data["shannon_radius_A"]
        if radius is None:
            radius = data["ionic_radius_A"]
        if radius is None:
            raise ValueError("Нет радиуса Шеннона и стандартного ионного радиуса")
        return float(radius)

    return ionic_radius
