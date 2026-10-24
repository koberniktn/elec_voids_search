import numpy as np
from ase.io import write

from gridmlip import Grid

from .void_search import clustered_local_minima # see void_search.py file

from pymatgen.core import Structure
from .candidate_tables import candidate_table
from .config import DEFAULT_CONFIG, resolve_config


def generate_grid(file: str, specie: int, **GRID_PARAMS):
    g = Grid.from_file(file, specie, **GRID_PARAMS)
    g.generate_mesh()

    distances_matrix = g.min_dists[g.mapping].reshape(g.mesh_shape)
    g.data = distances_matrix

    return g, distances_matrix


def find_voids(distances_matrix, g, debug=False, debug_file=None,
               minima_window=DEFAULT_CONFIG.grid_minima_window,
               relative_max_value=None,
               verbose=DEFAULT_CONFIG.grid_verbose):
    E = -distances_matrix # multiplied by -1 because we want to find local minima


    clusters = clustered_local_minima(E,
                              min_distance=None,       # minimum distance between voids
                              relative_max_value=relative_max_value, # None keeps every local maximum
                              cell = g.atoms.cell,
                              size = minima_window
                             )

    base = g.atoms.copy()
    cluster_frac_positions = []
    for i, v, n in clusters:
        if verbose:
            print(i, v, n)
        scaled_site = i/(np.array(E.shape) - 1)
        base.append('X')
        base.positions[-1] = base.cell.cartesian_positions(scaled_site)
        cluster_frac_positions.append(scaled_site)

    if debug:
        write(debug_file, base)

    return cluster_frac_positions


def grid_candidates(path, config=None, **overrides):
    """Table adapter for the grid-search blocks from test_easy.ipynb."""
    settings = resolve_config(config, **overrides)
    GRID_PARAMS = settings.grid_params()

    g, distances_matrix = generate_grid(str(path), settings.probe_atomic_number, **GRID_PARAMS)
    if not np.isfinite(distances_matrix).all():
        raise ValueError("Grid distances are not finite; increase grid_search_cutoff_A")
    cluster_frac_positions = find_voids(
        distances_matrix, g,
        minima_window=settings.grid_minima_window,
        relative_max_value=None,
        verbose=settings.grid_verbose,
    )

    structure = Structure.from_file(path)
    table = candidate_table(structure, cluster_frac_positions)
    shape = np.array(distances_matrix.shape) - 1
    parameters = {
        "GRID_PARAMS": GRID_PARAMS,
        "mesh_shape_core": shape.tolist(),
        "n_grid_samples": int(np.prod(shape)),
        "n_grid_candidates": len(table),
        "resolution_A": GRID_PARAMS["resolution"],
        "min_distance": None,
        "relative_max_value": None,
        "minima_window": settings.grid_minima_window,
        "probe_atomic_number": settings.probe_atomic_number,
    }
    return table, parameters
