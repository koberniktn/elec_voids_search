"""Legacy helper for historical audits; not used by the current search pipeline."""
import numpy as np
import pandas as pd
from ..symmetry.periodic_groups import periodic_groups


def final_unique_candidates(structure, passed, analyzer, tolerance):
    """Recheck symmetry at final coordinates; select the largest fit margin.

    Operates on passed candidates only: indeterminate/rejected points cannot
    bridge two passing groups. The original evaluated tables remain unchanged.
    """
    table = passed.reset_index(drop=True).copy()
    groups = periodic_groups(structure, table[['frac_x', 'frac_y', 'frac_z']],
                             tolerance, analyzer.get_symmetry_operations(cartesian=False))
    rows = []
    for group_id, indices in enumerate(groups):
        margins = table.loc[indices, 'geometry_margin_A'].to_numpy(dtype=float)
        chosen = indices[int(np.argmax(margins))]
        row = table.iloc[chosen].to_dict()
        row.update(final_group_id=group_id, final_member_count=len(indices),
                   final_member_candidate_ids=[int(i) for i in table.loc[indices, 'candidate_id']],
                   final_member_symmetry_group_ids=[int(i) for i in table.loc[indices, 'symmetry_group_id']])
        rows.append(row)
    columns = list(table.columns) + ['final_group_id', 'final_member_count',
                                    'final_member_candidate_ids', 'final_member_symmetry_group_ids']
    return pd.DataFrame(rows, columns=columns)
