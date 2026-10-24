import numpy as np
import pandas as pd


def candidate_table(structure, frac):
    frac = np.asarray(frac, dtype=float).reshape((-1, 3)) % 1.
    cart = frac @ structure.lattice.matrix
    result = pd.DataFrame(frac, columns=["frac_x", "frac_y", "frac_z"])
    result.insert(0, "candidate_id", np.arange(len(result)))
    result[["cart_x_A", "cart_y_A", "cart_z_A"]] = cart
    result["nearest_atom_center_A"] = structure.lattice.get_all_distances(frac, structure.frac_coords).min(axis=1) if len(frac) else []
    return result
