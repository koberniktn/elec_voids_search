from pathlib import Path

from pymatgen.core import DummySpecies, Structure
from pymatgen.io.cif import CifWriter


def export_candidates(structure, table, directory, method, parameters, write_cif=True):
    """CSV/JSON coordinates and host+X CIFs; X are markers, not inserted Li."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    table.to_csv(directory / f"{method}_candidates.csv", index=False)

    import json
    payload = dict(method=method, parameters=parameters,
                   coordinate_system="fractional coordinates of the input cell; Cartesian in angstrom",
                   lattice_matrix_A=structure.lattice.matrix.tolist(),
                   host=structure.as_dict(), candidates=json.loads(table.to_json(orient="records", double_precision=15)))
    (directory / f"{method}_candidates.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    if write_cif and len(table):
        frac = table[["frac_x", "frac_y", "frac_z"]].to_numpy()
        markers = Structure(structure.lattice, [DummySpecies("X")]*len(frac), frac)
        CifWriter(markers, symprec=None).write_file(directory / f"{method}_markers.cif")
        combined = structure.copy()
        
        for point in frac:
            combined.append(DummySpecies("X"), point)
        CifWriter(combined, symprec=None).write_file(directory / f"{method}_host_and_markers.cif")
