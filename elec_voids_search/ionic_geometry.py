"""Manual oxidation states and complete, auditable per-site/per-pore radii."""
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Species

from .shannon_radii import crystalnn_finder
from .site_atomic_data import get_site_atomic_data, get_cn_for_ion, get_shannon_radius
from .radius_resolution import resolve_ionic_radius, RadiusResolutionError


def validate_oxidation_states(structure, settings):
    """Require explicit charges for all host elements; never invoke BVAnalyzer."""
    if not len(structure) or not structure.is_ordered:
        raise ValueError("Geometry requires a nonempty ordered structure")
    elements = {site.specie.symbol for site in structure}
    missing = elements - set(settings.geometry_oxidation_states)
    if missing:
        raise ValueError("Supply geometry_oxidation_states for all host elements. Missing: "
                         + ", ".join(sorted(missing)))


def prepare_ionic_geometry(structure, settings):
    """Original site functions operate on the explicitly charged input copy."""
    validate_oxidation_states(structure, settings)
    charged = structure.copy()
    charged.add_oxidation_state_by_element(settings.geometry_oxidation_states)
    site_data, warning_text = {}, ""
    try:
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            charged, site_data = get_site_atomic_data(
                charged, cutoff=settings.geometry_neighbor_cutoff_A, spin=settings.geometry_spin
            )
        warning_text = "; ".join(str(message.message) for message in messages)
    except (ValueError, RuntimeError, KeyError) as error:
        warning_text = str(error)
    finder = crystalnn_finder(settings.geometry_neighbor_cutoff_A)
    rows = []
    for index, site in enumerate(charged):
        issue = ""
        data = site_data.get(index)
        if data is None:
            data = dict(element=site.specie.symbol, coordination=None,
                        oxidation_state=site.specie.oxi_state, atomic_radius_A=None,
                        ionic_radius_A=site.specie.element.ionic_radii.get(site.specie.oxi_state),
                        shannon_radius_A=None)
            try:
                data["coordination"] = get_cn_for_ion(index, charged, finder=finder)
                data["shannon_radius_A"] = get_shannon_radius(
                    site.specie, data["coordination"], spin=settings.geometry_spin
                )
            except (ValueError, RuntimeError, KeyError) as error:
                issue = str(error)
        override = settings.geometry_host_radius_overrides_A.get(
            str(index), settings.geometry_host_radius_overrides_A.get(site.specie.symbol)
        )
        try:
            resolved = resolve_ionic_radius(site.specie.symbol, site.specie.oxi_state,
                data["coordination"], settings.geometry_spin,
                settings.geometry_allow_ionic_fallback, override)
        except RadiusResolutionError as error:
            raise RadiusResolutionError(f"Host site {index}: {error}") from error
        rows.append(dict(site_index=index, **data, **resolved,
                         oxidation_source="user", radius_issue=issue, radius_warning=warning_text))
    return charged, pd.DataFrame(rows)


class ProbeRadiusCalculator:
    """Insert one probe into a copy of the original periodic cell at a time."""

    def __init__(self, structure, settings):
        validate_oxidation_states(structure, settings)
        self.settings = settings
        self.probe_structure = structure.copy()
        self.probe_structure.add_oxidation_state_by_element(settings.geometry_oxidation_states)
        self.cache = {}

    def __call__(self, frac_point):
        key = tuple(np.asarray(frac_point, dtype=float) % 1.)
        if key in self.cache:
            return self.cache[key].copy()
        settings = self.settings
        working = self.probe_structure.copy()
        working.append(Species(settings.probe_element, settings.probe_oxidation_state),
                       np.asarray(frac_point))
        index = len(working) - 1
        cn, issue = None, ""
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always")
            try:
                cn = get_cn_for_ion(index, working, cutoff=settings.geometry_neighbor_cutoff_A)
            except (ValueError, RuntimeError, KeyError) as error:
                issue = str(error)
            resolved = resolve_ionic_radius(settings.probe_element, settings.probe_oxidation_state,
                cn, settings.geometry_probe_spin, settings.geometry_allow_ionic_fallback,
                settings.geometry_probe_radius_override_A)
        result = dict(probe_element=settings.probe_element,
                      probe_oxidation_state=settings.probe_oxidation_state,
                      probe_coordination=cn, probe_radius_issue=issue,
                      probe_radius_warning="; ".join(str(message.message) for message in messages))
        result.update({"probe_" + key: value for key, value in resolved.items()})
        self.cache[key] = result
        return result.copy()


def occupied_probe_controls(structure, sites, settings):
    """Native occupied-ion controls, excluding only the central site."""
    radii = sites.radius_A.to_numpy(dtype=float)
    rows = []
    tolerance = settings.geometry_overlap_tolerance_A + settings.geometry_numeric_tolerance_A
    for index, site in enumerate(structure):
        if site.specie.symbol != settings.probe_element:
            continue
        charge = getattr(site.specie, "oxi_state", None)
        if charge != settings.probe_oxidation_state:
            continue
        probe_radius = radii[index]
        neighbors = structure.get_neighbors(site, settings.geometry_neighbor_cutoff_A)
        clearances = [float(n.nn_distance) - radii[int(n.index)] for n in neighbors
                      if np.isfinite(radii[int(n.index)])]
        free_radius = min(clearances, default=np.nan)
        required = settings.geometry_minimum_fit_fraction * probe_radius
        margin = free_radius - required
        complete = (np.isfinite(radii).all() and np.isfinite(probe_radius)
                    and max(radii) + required <= settings.geometry_neighbor_cutoff_A)
        status = ("rejected" if np.isfinite(margin) and margin < -tolerance
                  else "passed" if complete and np.isfinite(margin) else "indeterminate")
        rows.append(dict(site_index=index, probe_element=settings.probe_element,
                         probe_oxidation_state=charge, probe_coordination=sites.iloc[index].coordination,
                         probe_radius_A=probe_radius, free_radius_A=free_radius,
                         geometry_margin_A=margin, geometry_status=status,
                         frac_x=site.frac_coords[0], frac_y=site.frac_coords[1], frac_z=site.frac_coords[2]))
    return pd.DataFrame(rows, columns=["site_index", "probe_element", "probe_oxidation_state",
        "probe_coordination", "probe_radius_A", "free_radius_A", "geometry_margin_A",
        "geometry_status", "frac_x", "frac_y", "frac_z"])
