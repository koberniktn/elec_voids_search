"""Resolve a finite radius without changing oxidation state or radius convention."""
from functools import lru_cache
import math

from pymatgen.core import Species
from .site_atomic_data import get_shannon_radius

ROMAN_CN = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI',
            7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X', 11: 'XI', 12: 'XII'}


class RadiusResolutionError(ValueError):
    """A user radius is required; never substitute a different charge or NaN."""


def _positive(value):
    return value is not None and math.isfinite(float(value)) and float(value) > 0


@lru_cache(maxsize=None)
def _resolve(element, charge, coordination, spin, allow_ionic, override):
    specie = Species(element, charge)
    if override is not None:
        return float(override), 'user', (), (), ()
    if coordination is not None:
        exact = get_shannon_radius(specie, coordination, spin=spin)
        if _positive(exact):
            return float(exact), 'shannon_exact', (coordination,), (float(exact),), ()
    table = specie.element.data.get('Shannon radii', {}).get(str(int(charge)), {})
    available = {}
    spins = {}
    for cn, roman in ROMAN_CN.items():
        records = table.get(roman, {})
        # Geometric variants (IVSQ, IIIPY, ...) are not inferred from CN alone.
        if spin in records:
            selected_spin = spin
        elif len(records) == 1:
            selected_spin = next(iter(records))
        else:
            continue
        value = records[selected_spin].get('ionic_radius')
        if _positive(value):
            available[cn] = float(value)
            spins[cn] = selected_spin
    if coordination is not None and available:
        lower = [cn for cn in available if cn < coordination]
        upper = [cn for cn in available if cn > coordination]
        if lower and upper:
            lo, hi = max(lower), min(upper)
            # Do not interpolate across different spin assignments.
            if spins[lo] == spins[hi]:
                radius = available[lo] + (available[hi] - available[lo]) * (coordination - lo) / (hi - lo)
                return radius, 'shannon_interpolated', (lo, hi), (available[lo], available[hi]), (spins[lo], spins[hi])
        nearest = min(available, key=lambda cn: (abs(cn - coordination), cn))
        return available[nearest], 'shannon_nearest_cn', (nearest,), (available[nearest],), (spins[nearest],)
    if allow_ionic:
        # Species.ionic_radius can average other charges when data are missing.
        # Require a record for this exact oxidation state instead.
        ionic = specie.element.ionic_radii.get(charge)
        if _positive(ionic):
            return float(ionic), 'pymatgen_charge_specific_ionic', (), (), ()
    raise RadiusResolutionError(
        f'No usable radius for {element}{charge:+g}, CN={coordination}, spin={spin!r}. '
        'Supply geometry_host_radius_overrides_A or geometry_probe_radius_override_A; '
        'check the specified oxidation state/spin. No unresolved radius is exported.'
    )


def resolve_ionic_radius(element, charge, coordination, spin='', allow_ionic=True, override=None):
    if not math.isfinite(float(charge)) or not float(charge).is_integer():
        raise ValueError("Radius lookup requires the explicitly supplied integer oxidation state")
    if override is not None and not _positive(override):
        raise ValueError("Radius override must be finite and positive")
    radius, source, references, reference_radii, reference_spins = _resolve(
        element, charge, coordination, spin, allow_ionic, override
    )
    return dict(radius_A=radius, radius_source=source,
                radius_requested_coordination=coordination,
                radius_reference_coordinations=list(references),
                radius_reference_values_A=list(reference_radii),
                radius_reference_spins=list(reference_spins),
                radius_approximate=source not in {'shannon_exact', 'user'},
                radius_convention='Shannon effective ionic' if source.startswith('shannon') else source)
