"""Create a scaled nanohelix test system.

This script is a small test runner that builds a nanohelix by
calling the Construction package helpers. It selects scaling presets
and writes duplicated/interior lattice files used for system tests.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Construction.import_libraries import *
from Construction.distortion import *
from Construction.analysis import *
from Construction.read_write import *

# Physical type and mass maps used by the test harness
type_map = {"Si": 1, "O": 2}
mass_map = {1: 28.0855, 2: 15.9994}

# ---------------------------------------------------------------------------
# Nanohelix reference parameters (real system)
# ---------------------------------------------------------------------------
rota = 1.0
D = 244
P = 453
T = 112
W = 226


# ---------------------------------------------------------------------------
# Scaling presets derived proportionally from the reference geometry
# ---------------------------------------------------------------------------
Pitch_list = [10, 120, 150, 200, 400]
print("P=", Pitch_list)
D_list = [(p / P) * D for p in Pitch_list]
print("D=", D_list)
Thickness_list = [(p / P) * T for p in Pitch_list]
print("T=", Thickness_list)
Width_list = [(p / P) * W for p in Pitch_list]
print("W=", Width_list)
Int_thick_list = [10, 10, 10, 15, 20]

# Index of the preset to use
a = 3


# ---------------------------------------------------------------------------
# Selected system parameters (preset)
# ---------------------------------------------------------------------------
diameter = D_list[a]
width = Width_list[a]
pitch = Pitch_list[a]
thickness = Thickness_list[a]
int_thick = Int_thick_list[a]

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "files_lattices", "Model_file", "beta_quartz.data")

# ---------------------------------------------------------------------------
# System build
# ---------------------------------------------------------------------------

(Pos_transfo, Types, Lims_tot,
 Angles_OH, Pos_transfo_int, Types_int,
 Lims_tot_int) = create_syst(
    rota,
    diameter,
    pitch,
    width,
    thickness,
    int_thick,
    Twist=False,
    do_clean=True,
    file_duplicate=file_path,
    file_output="quartz_dupl.data",
    file_output_cast="quartz_int.data",
    mass_map=mass_map,
)

