import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Construction.import_libraries import *
from Construction.distortion import *
from Construction.analysis import *
from Construction.read_write import *
from Construction.add_chromophore import *

### SiO2 ###
target_folder = 'sio2_pitch_200'


# ---------------------------------------------------------------------------
# Nanohelix reference parameters (silica real system)
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
Int_thick_list = [0, 0, 0, 0, 0] # No cast



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


original_dir = os.getcwd()
os.chdir(target_folder)
print("Entering target folder", target_folder)

file = "last_timestep.lammpstrj"
list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file,unscale=True)

# Analyze data file 
list_TSTEP=[0]
list_Pos = list_ATOMS[:,:,2:]
list_Types = list_ATOMS[:,:,1]
Pos = list_ATOMS[-1][:,2:]
Types = list_ATOMS[-1][:,1]
Lims = list_BOX[-1]

### Number of atoms ###
n_total = len(Pos)
print("Total number of atoms:", n_total)


OH_atoms = np.where(((Types == 3) |(Types == 4)))[0]
Pos_OH = Pos[OH_atoms]

dz = 1.0
### Atoms on the contour of the surface ###
Pos_contour, contour_indices_global, contours_per_slice = external_surface(Pos, list_BOX, dz=dz)

visualize_slice_with_circles(
    Pos, 
    10, 
    dz=1.0
)


### Number of atoms on the surface ###
n_total_surf = len(Pos_contour)
print("Total number of atoms on the surface:", n_total_surf)

### Total number of Si_atoms  and O_atoms ###
## Si ##
Types_contour = Types[contour_indices_global]
Si_atoms_contour = np.where((Types_contour == 1))[0]
n_Si_atoms = len(Si_atoms_contour)
print("Total number of Si atoms:", len(Si_atoms_contour))

## O ##
O_atoms_contour_type2 = np.where((Types_contour == 2))[0]
O_atoms_contour_type3 = np.where((Types_contour == 3))[0]
n_O_atoms = len(O_atoms_contour_type2) + len(O_atoms_contour_type3)
print("Total number of O atoms:", n_O_atoms)

### Only insturated ###
D, Si_count_O, O_count_Si =  compute_hist_neighbors(Pos_contour,Types_contour,cube=30,threshold_type1=2,threshold_type2=2,rdf_max=5)

## Si ##
n_Si_atoms_insat = 0
for Si in Si_count_O:
    if Si == 3:
        n_Si_atoms_insat +=1

print("Si atoms insaturated:", n_Si_atoms_insat)

## O ##
n_O_atoms_insat = 0
for O in O_count_Si:
    if O == 1:
        n_O_atoms_insat +=1
print("O atoms insaturated:", n_O_atoms_insat)

H_atoms_contour = np.where((Types_contour == 4))[0]
n_H_atoms = len(H_atoms_contour)
print("Total number of H atoms:", n_H_atoms)

print(np.unique(Types))

mol_file = "1-propanamine.sdf"
type_map = {"C": 5, "H": 6, "N": 7}
Pos_contour, contour_indices_global, contour_per_slice = external_surface(Pos, list_BOX, dz=1.0, n_points=1000, twist=False)

# ---------------------------------------------------------------------------
# Compute surface areas using different methods
# ---------------------------------------------------------------------------
surface_area_method1 = compute_surface_method_1(n_Si_atoms, n_O_atoms, n_H_atoms) * 10**(-2)
surface_area_method2 = compute_surface_method_2(contour_per_slice, dz=1.0) * 10**(-2)
surface_area_method3 = compute_surface_method_3(Pos, list_BOX, dz=1.0, n_points=1000) * 10**(-2)
surface_area_method4 = compute_surface_method_4(radius=diameter / 2, pitch=pitch) * 10**(-2)
surface_area_method5 = compute_surface_method_5(
    Pos, diameter, width, thickness, pitch, list_BOX,
    n_x=100, n_z=100, n_y=100, n=10, circling=True, face='outer'
) * 10**(-2)

# Display surface area summary table
print("\n" + "=" * 80)
print("SURFACE AREA CALCULATION SUMMARY")
print("=" * 80)
print(f"{'Method':<35} {'Surface Area (nm²)':<20} {'Description'}")
print("-" * 80)
print(f"{'Method 1: Van der Waals':<35} {surface_area_method1:>18.4f} {'Atom count based'}")
print(f"{'Method 2: Slice Perimeter':<35} {surface_area_method2:>18.4f} {'Contour integration'}")
print(f"{'Method 3: Circle Perimeter':<35} {surface_area_method3:>18.4f} {'Circle difference'}")
print(f"{'Method 4: Tube Formula':<35} {surface_area_method4:>18.4f} {'Tubular surface'}")
print(f"{'Method 5: Transformed Cuboid':<35} {surface_area_method5:>18.4f} {'Mesh-based sampling'}")
print("=" * 80)

# Default to method 5 for downstream calculations
surface_area_nm2 = surface_area_method3
print(f"\nSelected surface area: {surface_area_nm2:.4f} nm² (Method 5)")
print()