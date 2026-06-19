"""
File reading and writting utilities for SHAnC.

This module contains readers, writers, and helpers for XYZ and LAMMPS
structure files, type/mass mapping, and box parsing. It is designed to
support frame-by-frame simulation data extraction and conversion to the
internal arrays used by the SHAnC analysis pipeline.

Functions include:
    read_xyz, write_xyz, read_dump, write_dump, read_data, write_data
    plus auxiliary format parsers for type map and box information.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Construction.import_libraries import *


# ---------------------------------------------------------------------------
# Default atomic masses keyed by element symbol
# ---------------------------------------------------------------------------

ELEMENT_MASSES = {
    "H": 1.0080, "He": 4.0026,
    "Li": 6.941, "Be": 9.0122,
    "C": 12.011, "N": 14.007, "O": 15.9994,
    "F": 18.998, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.0855, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "K": 39.098,
    "Ca": 40.078, "Ti": 47.867, "Cr": 51.996,
    "Fe": 55.845, "Ni": 58.693, "Cu": 63.546,
    "Zn": 65.38, "Ga": 69.723, "Ge": 72.63,
    "Zr": 91.224, "Ag": 107.868, "Sn": 118.710,
    "Pt": 195.084, "Au": 196.967, "Pb": 207.2,
}

def _mass_to_symbol_map(mass_map):
    """
    Build {type_id: element_symbol} from a {type_id: mass} dict by finding
    the nearest match in ELEMENT_MASSES. Falls back to "X<id>" when no
    element is within 0.5 Da.
    """
    symbol_map = {}
    for tid, mass in mass_map.items():
        best_sym, best_diff = None, float("inf")
        for sym, m in ELEMENT_MASSES.items():
            diff = abs(m - mass)
            if diff < best_diff:
                best_diff, best_sym = diff, sym
        symbol_map[int(tid)] = best_sym if best_diff < 0.5 else "X{}".format(tid)
    return symbol_map


# ---------------------------------------------------------------------------
# XYZ files
# ---------------------------------------------------------------------------


def read_xyz(file_name, type_map=None, metal=False):
    """Parse an XYZ file into timestep, box, and atom arrays.

    This reader supports XYZ frames with optional LAMMPS-style timestep and
    lattice information in the comment line. If `metal=True`, the file is
    assumed to contain a periodic metal cell and the coordinates are
    normalized into a half-open cubic cell.

    Parameters
    ----------
    file_name : str
        Path to the XYZ file.
    type_map : dict, optional
        Optional mapping from element symbol to type id.
    metal : bool, optional
        Enable FCC metal cell inference.

    Returns
    -------
    list_TSTEP : list
        Timesteps for each frame.
    list_NUM_AT : list
        Atom counts per frame.
    list_BOX : list
        Box limits per frame.
    list_ATOMS : ndarray
        Atom records per frame.
    """
    with open(file_name, "r") as f:
        lines = f.readlines()

    list_TSTEP = []
    list_NUM_AT = []
    list_BOX = []
    list_ATOMS = []

    auto_type_map = {} if type_map is None else dict(type_map)
    next_type_id = max(auto_type_map.values(), default=0) + 1

    # Parse frames from the XYZ file one timestep at a time.
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        num_at = int(line)
        n_lines_to_skip = num_at # save BEFORE num_at is modified below
        comment = lines[i + 1].strip() if (i + 1) < len(lines) else ""

        # ---- parse comment line ----
        tstep = 0
        box = []
        lx = ly = lz = 1.0

        if comment.lower().startswith("timestep"):
            parts = comment.split()
            tstep = int(parts[1])
            lx, ly, lz = float(parts[2]), float(parts[3]), float(parts[4])
            box = [[0.0, lx], [0.0, ly], [0.0, lz]]

        elif "Lattice=" in comment or 'Lattice="' in comment:
            m = re.search(r'[Ll]attice=["\']?([^"\']+)["\']?', comment)
            if m:
                vals = list(map(float, m.group(1).split()))
                lx, ly, lz = vals[0], vals[4], vals[8]
                box = [[0.0, lx], [0.0, ly], [0.0, lz]]
                try:
                    m2 = re.search(r'[Tt]ime[Ss]tep=(\d+)', comment)
                    if m2:
                        tstep = int(m2.group(1))
                except Exception:
                    pass

        # ---- read atoms always store raw coords here ----
        atoms = []
        positions = []
        for j in range(num_at):
            ls = lines[i + 2 + j].split()
            elem = ls[0]
            x, y, z = float(ls[1]), float(ls[2]), float(ls[3])

            if elem not in auto_type_map:
                auto_type_map[elem] = next_type_id
                next_type_id += 1
            type_at = auto_type_map[elem]

            atoms.append([j + 1, type_at, x, y, z]) # always raw
            positions.append([x, y, z])

        positions = np.array(positions)

        # ---- infer box AFTER reading all atoms ----
        # If the frame header did not include box dimensions, infer a cubic
        # lattice for metal systems using the nearest-neighbour distance.
        if not box:
            if metal:
                # d_NN = a / sqrt(2) for FCC 
                D = sd.cdist(positions, positions)
                np.fill_diagonal(D, np.inf)
                d_NN = np.min(D)
                a = d_NN * np.sqrt(2)
                origin = np.min(positions, axis=0)
                positions = positions - origin
                positions = positions % a
                for k in range(len(atoms)):
                    atoms[k][2] = positions[k, 0]
                    atoms[k][3] = positions[k, 1]
                    atoms[k][4] = positions[k, 2]

                box = [[0.0, a], [0.0, a], [0.0, a]]

                # ---- FIX: remove periodic image atoms on the top faces ----
                # The XYZ contains atoms at x=0 AND x=a (same for y, z).
                # When tiling, cell k's top face = cell k+1's bottom face -> duplicates.
                # Keeping only the half-open cell [0, a) fixes this at the source.
                tol = 1e-3 # robust against floating point
                atoms_clean = []
                for atom in atoms:
                    x, y, z = atom[2], atom[3], atom[4]
                    on_top_face = (x > a - tol) or (y > a - tol) or (z > a - tol)
                    if not on_top_face:
                        atoms_clean.append(atom)

                n_removed = len(atoms) - len(atoms_clean)
                print(f"Metal half-open cell: removed {n_removed} top-face atoms (a={a:.4f} Å, d_NN={d_NN:.4f} Å)")
                print(f" Unit cell: {len(atoms_clean)} atoms (was {len(atoms)})")
                atoms = atoms_clean
                num_at = len(atoms_clean)
            else:
                box = []

        list_TSTEP.append(tstep)
        list_NUM_AT.append(num_at)
        list_BOX.append(box)
        list_ATOMS.append(atoms)
        i += 2 + n_lines_to_skip

    return list_TSTEP, list_NUM_AT, list_BOX, np.array(list_ATOMS, dtype=object)

def write_xyz(file_out, list_TSTEP, list_BOX, list_ATOMS, symbol_map, last_only=False):
    """
    Shared XYZ writer used by both convert_* functions.
    Positions in list_ATOMS must already be Cartesian (not fractional).
    The comment line carries timestep + box info so the file can be
    round-tripped via read_xyz.
    """
    frames = list(zip(list_TSTEP, list_BOX, list_ATOMS))
    if last_only:
        frames = [frames[-1]]

    # Write each frame to the output XYZ file. The comment line includes
    # timestep and box dimensions so the format can be read back by read_xyz.
    with open(file_out, "w") as f:
        for tstep, box, atoms in frames:
            atoms = np.asarray(atoms)
            f.write("{}\n".format(len(atoms)))

            if box:
                lx = box[0][1] - box[0][0]
                ly = box[1][1] - box[1][0]
                lz = box[2][1] - box[2][0]
                f.write("timestep {} {:3.6f} {:3.6f} {:3.6f}\n".format(tstep, lx, ly, lz))
            else:
                f.write("timestep {}\n".format(tstep))

            for row in atoms:
                tid = int(row[1])
                sym = symbol_map.get(tid, "X{}".format(tid))
                f.write("{} {:3.6f} {:3.6f} {:3.6f}\n".format(sym, row[2], row[3], row[4]))

    print("Written: {} ({} frame{})".format(file_out, len(frames), "s" if len(frames) > 1 else ""))



# ---------------------------------------------------------------------------
# LAMMPS output files
# ---------------------------------------------------------------------------


def read_dump(dump_file, unscale=False):
    """Read a LAMMPS dump file."""
    # The LAMMPS dump file is scanned line-by-line using state flags for each
    # ITEM: section. Atom coordinates may be unscaled into Cartesian units.
    flag_step = 0
    flag_num_at = 0
    flag_box_bound = 0
    flag_atoms = 0
    list_TSTEP = []
    list_NUM_AT = []
    list_BOX = []
    list_ATOMS = []
    BOX = []
    list_at_t = []

    for line in open(dump_file, "r"):
        if flag_step:
            list_TSTEP.append(int(line))
            flag_step = 0
        elif flag_num_at:
            list_NUM_AT.append(int(line))
            flag_num_at = 0
        elif flag_box_bound:
            lsplit = line.split()
            BOX.append([float(lsplit[0]), float(lsplit[1])])
            flag_box_bound -= 1
            if not flag_box_bound:
                list_BOX.append(BOX)
        elif flag_atoms:
            if "ITEM: TIMESTEP" in line:
                flag_step = 1
                flag_atoms = 0
                list_ATOMS.append(list_at_t)
                list_at_t = []
            else:
                lsplit = line.split()
                if len(lsplit) != 5:
                    raise TypeError("Expected dump with: id type xs ys zs")
                if unscale:
                    Lx = BOX[0][1] - BOX[0][0]
                    Ly = BOX[1][1] - BOX[1][0]
                    Lz = BOX[2][1] - BOX[2][0]
                    list_at_t.append([int(lsplit[0]), int(lsplit[1]),
                                      float(lsplit[2]) * Lx + BOX[0][0],
                                      float(lsplit[3]) * Ly + BOX[1][0],
                                      float(lsplit[4]) * Lz + BOX[2][0]])
                else:
                    list_at_t.append([int(lsplit[0]), int(lsplit[1]),
                                      float(lsplit[2]), float(lsplit[3]), float(lsplit[4])])
        elif "ITEM: TIMESTEP" in line:
            flag_step = 1
        elif "ITEM: NUMBER OF ATOMS" in line:
            flag_num_at = 1
        elif "ITEM: BOX BOUNDS" in line:
            flag_box_bound = 3
            BOX = []
        elif "ITEM: ATOMS" in line:
            flag_atoms = 1
            list_at_t = []

    list_ATOMS.append(list_at_t)

    try:
        list_ATOMS = np.array(list_ATOMS)
    except Exception:
        print("Atoms lost/removed during dynamics — extracting last timestep only.")
        list_TSTEP = [list_TSTEP[-1]]
        list_NUM_AT = [list_NUM_AT[-1]]
        list_BOX = [list_BOX[-1]]
        list_ATOMS = np.array([list_ATOMS[-1]])

    if unscale:
        Lx = BOX[0][1] - BOX[0][0]
        Ly = BOX[1][1] - BOX[1][0]
        Lz = BOX[2][1] - BOX[2][0]
        C_min = np.mean(list_ATOMS[:, :, 2:], axis=1) + np.array([Lx / 2, Ly / 2, Lz / 2])
        C_min[:, 2] = 0
        C_min = C_min.reshape((len(C_min), 1, 3))
        list_ATOMS[:, :, 2:] = list_ATOMS[:, :, 2:] - C_min
        list_ATOMS[:, :, 2] = list_ATOMS[:, :, 2] % Lx
        list_ATOMS[:, :, 3] = list_ATOMS[:, :, 3] % Ly
        list_ATOMS[:, :, 4] = (list_ATOMS[:, :, 4] - BOX[2][0]) % Lz + BOX[2][0]

    return list_TSTEP, list_NUM_AT, list_BOX, np.array(list_ATOMS)

def write_dump(file_name, list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS):
    """Write a LAMMPS dump file."""
    # Output dump data in the standard LAMMPS ITEM: TM structure format.
    with open(file_name, "w") as file:
        for tstep, num_at, box, atoms in zip(list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS):
            file.write("ITEM: TIMESTEP\n{}\n".format(tstep))
            file.write("ITEM: NUMBER OF ATOMS\n{}\n".format(num_at))
            file.write("ITEM: BOX BOUNDS pp pp pp\n")
            for box_c in box:
                file.write("{:3.6f} {:3.6f}\n".format(box_c[0], box_c[1]))
            file.write("ITEM: ATOMS id type xs ys zs\n")
            for at in atoms:
                file.write("{} {} {:3.6f} {:3.6f} {:3.6f}\n".format(
                    int(at[0]), int(at[1]), at[2], at[3], at[4]))

# ---------------------------------------------------------------------------
# LAMMPS input files
# ---------------------------------------------------------------------------

def read_data(file, do_scale=True, atom_style="full"):
    """Read a LAMMPS data file."""
    # Support both 'full' LAMMPS format (with charges/coordinates) and the
    # simpler 'atom' style used by this project for structure generation.
    if atom_style == "full":
        BOX = []
        list_at_t = []
        with open(file, "r") as f:
            for line in f:
                lsplit = line.split()
                if not lsplit:
                    continue
                if "xlo" in line or "ylo" in line or "zlo" in line:
                    BOX.append([float(lsplit[0]), float(lsplit[1])])
                if len(lsplit) == 7:
                    Lx = BOX[0][1] - BOX[0][0]
                    Ly = BOX[1][1] - BOX[1][0]
                    Lz = BOX[2][1] - BOX[2][0]
                    if do_scale:
                        list_at_t.append([int(lsplit[0]), int(lsplit[2]),
                                          float(lsplit[4]) / Lx,
                                          float(lsplit[5]) / Ly,
                                          float(lsplit[6]) / Lz])
                    else:
                        list_at_t.append([int(lsplit[0]), int(lsplit[2]),
                                          float(lsplit[4]), float(lsplit[5]), float(lsplit[6])])
                elif len(lsplit) == 6:
                    Lx = BOX[0][1] - BOX[0][0]
                    Ly = BOX[1][1] - BOX[1][0]
                    Lz = BOX[2][1] - BOX[2][0]
                    if do_scale:
                        list_at_t.append([int(lsplit[0]), float(lsplit[1]),
                                          float(lsplit[3]) / Lx,
                                          float(lsplit[4]) / Ly,
                                          float(lsplit[5]) / Lz])
                    else:
                        list_at_t.append([int(lsplit[0]), float(lsplit[1]),
                                          float(lsplit[3]), float(lsplit[4]), float(lsplit[5])])
        return [BOX], np.array([list_at_t])

    elif atom_style == "atom":
        Lims = []
        Atom_types = []
        Atom_pos = []
        mass_map = {}
        in_masses = False
        in_atoms = False

        # Parse atom-style LAMMPS data files: box bounds, type/mass blocks, and
        # atom coordinate lines. Box bounds may be inferred if missing.
        with open(file, "r"):
            for line in open(file):
                lsplit = line.split()
                if not lsplit:
                    continue

                if lsplit[0] == "Masses":
                    in_masses = True
                    in_atoms = False
                    continue
                if lsplit[0] == "Atoms":
                    in_masses = False
                    in_atoms = True
                    continue
                if in_masses and len(lsplit) >= 1 and lsplit[0].isalpha():
                    in_masses = False

                if len(lsplit) >= 4 and lsplit[2] in ("xlo", "ylo", "zlo"):
                    try:
                        Lims.append([float(lsplit[0]), float(lsplit[1])])
                    except ValueError:
                        pass
                    continue

                if in_masses:
                    try:
                        mass_map[int(lsplit[0])] = float(lsplit[1])
                    except (ValueError, IndexError):
                        pass
                    continue

                if in_atoms:
                    try:
                        if len(lsplit) == 5:
                            Atom_types.append(int(lsplit[1]))
                            Atom_pos.append([float(lsplit[2]), float(lsplit[3]), float(lsplit[4])])
                        elif len(lsplit) == 6:
                            Atom_types.append(int(lsplit[1]))
                            Atom_pos.append([float(lsplit[3]), float(lsplit[4]), float(lsplit[5])])
                        elif len(lsplit) == 7:
                            Atom_types.append(int(lsplit[2]))
                            Atom_pos.append([float(lsplit[4]), float(lsplit[5]), float(lsplit[6])])
                    except (ValueError, IndexError):
                        pass
                    continue

        Atom_pos = np.array(Atom_pos, dtype=float)

        if len(Lims) < 3:
            print("Warning: box bounds not found in '{}', inferring from atom positions.".format(file))
            if len(Atom_pos) > 1:
                from scipy.spatial.distance import cdist
                dists = cdist(Atom_pos, Atom_pos)
                np.fill_diagonal(dists, np.inf)
                padding = max(np.min(dists) / 2.0, 0.5)
            else:
                padding = 0.5
            Lims = [
                [np.min(Atom_pos[:, 0]) - padding, np.max(Atom_pos[:, 0]) + padding],
                [np.min(Atom_pos[:, 1]) - padding, np.max(Atom_pos[:, 1]) + padding],
                [np.min(Atom_pos[:, 2]) - padding, np.max(Atom_pos[:, 2]) + padding],
            ]

        z_min = np.min(Atom_pos[:, 2])
        Atom_pos = Atom_pos - np.array([0.0, 0.0, z_min])
        Lims[2][1] -= Lims[2][0]
        Lims[2][0] = 0.0

        return np.array(Lims), np.array(Atom_types), Atom_pos, mass_map

def write_data(file_name, Pos, Types, Lims, test_particle=False, Bonds_OH=[], Angles_OH=[], mass_map=None, Types_masses=None):
    """Write a LAMMPS data file."""
    # Write positions, types, optional OH bonds/angles, and mass definitions
    # in the atom-style LAMMPS data file format.
    _DEFAULT_TYPE_MASSES = {
        1: 28.0855, 2: 15.9994, 3: 15.9994, 4: 1.0080, 5: 28.0855, 6: 15.9994, 7: 1.0080,
    }
    effective_mass = dict(_DEFAULT_TYPE_MASSES)

    if Types_masses is not None:
        for entry in Types_masses:
            tokens = entry.split()
            try:
                effective_mass[int(tokens[0])] = float(tokens[1])
            except (ValueError, IndexError):
                pass

    if mass_map is not None:
        effective_mass.update(mass_map)

    unique_types = np.unique(Types)
    H_present = any(t in Types for t in (3, 4, 7))
    bonds = len(Bonds_OH) > 0
    angles = len(Angles_OH) > 0

    with open(file_name, "w") as file:
        file.write("\n")
        file.write("{} atoms\n".format(len(Types)))

        if bonds and H_present:
            file.write("{} bonds\n".format(len(Bonds_OH)))
        if angles and H_present:
            file.write("{} angles\n".format(len(Angles_OH)))
        file.write("\n")

        if test_particle:
            file.write("{} atom types\n".format(int(np.max(Types) - np.min(Types) + 1)))
        else:
            file.write("{} atom types\n".format(int(np.max(Types))))

        if bonds and H_present:
            file.write("1 bond types\n")
        if angles and H_present:
            file.write("1 angle types\n")

        file.write("\n")
        lo, hi = Lims[0]
        file.write("{:3.6f} {:3.6f} xlo xhi\n".format(lo, hi))
        lo, hi = Lims[1]
        file.write("{:3.6f} {:3.6f} ylo yhi\n".format(lo, hi))
        lo, hi = Lims[2]
        file.write("{:3.6f} {:3.6f} zlo zhi\n".format(lo, hi))
        file.write("\n")

        file.write("Masses\n\n")
        for t in sorted(unique_types):
            t = int(t)
            m = effective_mass.get(t, 1.0)
            if t not in effective_mass:
                print("Warning: no mass for type {}; using 1.0".format(t))
            file.write("{} {:3.6f}\n".format(t, m))
        file.write("\n")

        file.write("Atoms\n\n")
        for num, pos, typ in zip(range(len(Pos)), Pos, Types):
            file.write("{} 1 {} 0.0 {:3.6f} {:3.6f} {:3.6f}\n".format(
                num + 1, int(typ), pos[0], pos[1], pos[2]))

        if bonds and H_present:
            file.write("\nBonds\n\n")
            for num, bond in enumerate(Bonds_OH):
                file.write("{} 1 {} {}\n".format(num + 1, bond[0], bond[1]))

        if angles and H_present:
            file.write("\nAngles\n\n")
            for num, angle in enumerate(Angles_OH):
                file.write("{} 1 {} {} {}\n".format(num + 1, angle[0], angle[1], angle[2]))

    print("Written:", file_name)


# ---------------------------------------------------------------------------
# Conversion of files
# ---------------------------------------------------------------------------


def convert_data_to_xyz(file, last_only=False):
    """Convert a LAMMPS data file to XYZ format.

    Parameters
    ----------
    file : str
        Input LAMMPS data filename.
    last_only : bool, optional
        If True, write only the final frame.

    Returns
    -------
    Atom_pos : ndarray
        Atom positions from the converted file.
    Atom_types : ndarray
        Atom type ids.
    Lims : list
        Box limits.
    """
    Lims, Atom_types, Atom_pos, mass_map = read_data(file, do_scale=False, atom_style="atom")
    symbol_map = _mass_to_symbol_map(mass_map)

    atoms = np.column_stack([
        np.arange(1, len(Atom_types) + 1),
        Atom_types,
        Atom_pos,
    ])

    file_out = file.rsplit(".", 1)[0] + ".xyz"
    write_xyz(file_out, list_TSTEP=[0], list_BOX=[Lims.tolist()], list_ATOMS=[atoms], symbol_map=symbol_map)
    return Atom_pos, Atom_types, Lims

def convert_dump_to_xyz(file, last_only=False, data_file=None, type_map=None):
    """Convert a LAMMPS dump file to XYZ format.

    Parameters
    ----------
    file : str
        Input dump filename.
    last_only : bool, optional
        If True, write only the last timestep.
    data_file : str, optional
        Optional LAMMPS data file used to recover element type mapping.
    type_map : dict, optional
        Manual type_id to symbol mapping.

    Returns
    -------
    list_TSTEP : list
        Timesteps read from the dump.
    list_BOX : list
        Box limits for each frame.
    list_ATOMS : list
        Atom records for each frame.
    """
    print("Reading dump file:", file)
    list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file, unscale=True)

    symbol_map = {}
    if data_file is not None:
        _, _, _, mass_map = read_data(data_file, do_scale=False, atom_style="atom")
        symbol_map = _mass_to_symbol_map(mass_map)
    if type_map is not None:
        symbol_map.update({int(k): v for v, k in type_map.items()})

    file_out = file.rsplit(".", 1)[0] + ".xyz"
    write_xyz(file_out, list_TSTEP=list_TSTEP, list_BOX=list_BOX, list_ATOMS=list_ATOMS, symbol_map=symbol_map, last_only=last_only)
    return list_TSTEP, list_BOX, list_ATOMS


def wrap_and_deduplicate(Atom_pos, Lims, tol=1e-3):
    """Wrap positions into the periodic cell and remove duplicate atoms.

    Parameters
    ----------
    Atom_pos : array_like
        Atomic Cartesian positions.
    Lims : array_like
        Box limits [[lx,Lx],[ly,Ly],[lz,Lz]].
    tol : float, optional
        Distance tolerance for duplicate elimination.

    Returns
    -------
    ndarray
        Wrapped, deduplicated atom positions.
    """
    Atom_pos = np.asarray(Atom_pos, dtype=float)
    Lims = np.asarray(Lims, dtype=float)

    origin = Lims[:, 0]
    L = Lims[:, 1] - Lims[:, 0]

    frac = (Atom_pos - origin) / L
    frac_folded = frac % 1.0
    pos_wrapped = frac_folded * L

    for axis in range(3):
        near_top = np.abs(pos_wrapped[:, axis] - L[axis]) < tol
        pos_wrapped[near_top, axis] = 0.0

    grid = np.round(pos_wrapped / tol).astype(int)
    _, unique_idx = np.unique(grid, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)

    n_removed = len(Atom_pos) - len(unique_idx)
    if n_removed:
        print(f"[wrap_and_deduplicate] {n_removed} duplicate(s) removed ({len(Atom_pos)} → {len(unique_idx)} atoms)")
    else:
        print(f"[wrap_and_deduplicate] No duplicates found ({len(unique_idx)} atoms)")

    return pos_wrapped[unique_idx], unique_idx
