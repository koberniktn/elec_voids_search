from itertools import product

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi

from .candidate_tables import candidate_table
from ..config import DEFAULT_CONFIG, resolve_config


def find_voronoi_vertices(structure,
                          image_range=DEFAULT_CONFIG.voronoi_image_range,
                          boundary_tolerance=DEFAULT_CONFIG.voronoi_boundary_tolerance,
                          qhull_options=DEFAULT_CONFIG.voronoi_qhull_options):
    lattice_matrix = np.asarray(structure.lattice.matrix)
    original_frac = np.asarray(structure.frac_coords)

    repeated_cart = []
    images = tuple(range(-image_range, image_range + 1))
    for image in product(images, repeat=3):
        translated_frac = original_frac + np.asarray(image)
        repeated_cart.extend(translated_frac @ lattice_matrix)

    voronoi = Voronoi(np.asarray(repeated_cart), qhull_options=qhull_options)
    vertices_frac = voronoi.vertices @ np.linalg.inv(lattice_matrix)

    inside = np.all(vertices_frac >= -boundary_tolerance, axis=1)
    inside &= np.all(vertices_frac < 1.0 - boundary_tolerance, axis=1) # оператор &= старое и новое условие одновременно истина
    raw_vertices = np.mod(vertices_frac[inside], 1.0) # приводит к  диапозону [0, 1) по модулю 1.0

    raw_voronoi_df = pd.DataFrame(
        raw_vertices, columns=["frac_x", "frac_y", "frac_z"]
    )
    raw_voronoi_df.insert(0, "voronoi_id", range(len(raw_voronoi_df)))

    return raw_vertices, raw_voronoi_df


def voronoi_candidates(structure, config=None, return_details=False, **overrides):
    """All raw vertices; close sites are merged only after combining methods."""
    settings = resolve_config(config, **overrides)
    raw_vertices, raw_voronoi_df = find_voronoi_vertices(
        structure,
        image_range=settings.voronoi_image_range,
        boundary_tolerance=settings.voronoi_boundary_tolerance,
        qhull_options=settings.voronoi_qhull_options,
    )

    table = candidate_table(structure, raw_vertices)
    parameters = {
        "n_voronoi_raw_vertices": len(raw_vertices),
        "n_voronoi_candidates": len(table),
        "image_range": settings.voronoi_image_range,
        "n_periodic_images": (2 * settings.voronoi_image_range + 1) ** 3,
        "boundary_tolerance_fractional": settings.voronoi_boundary_tolerance,
        "qhull_options": settings.voronoi_qhull_options,
        "distance_merge_applied": False,
    }
    if return_details:
        return table, parameters, {
            "raw_vertices": raw_vertices,
            "raw_voronoi_df": raw_voronoi_df,
        }
    return table, parameters
