"""Occupied-site functions copied verbatim from strong_voids_screening.py."""
import pandas as pd


def site_has_element(site, symbol):
        return site_composition(site).get(symbol, 0.0) > 0


def site_composition(site):
    return {
        species.symbol: float(occupancy)
        for species, occupancy in site.species.items()
    }


def get_nonequivalent_occupied_sites(structure, analyzer, probe_element):
    symmetrized_structure = analyzer.get_symmetrized_structure()
    mobile_site_types = []

    for equivalent_group in symmetrized_structure.equivalent_indices:
        mobile_group = [
            i for i in equivalent_group
            if site_has_element(structure[i], probe_element)
        ]
        if mobile_group:
            mobile_site_types.append(mobile_group)

    mobile_types_df = pd.DataFrame([{
        "mobile_type": type_id + 1,
        "atom_indices": indices,
        "mobile_occupancies": [
            site_composition(structure[i]).get(probe_element, 0.0)
            for i in indices
        ],
        "multiplicity_in_cell": len(indices),
    } for type_id, indices in enumerate(mobile_site_types)])

    return mobile_types_df
