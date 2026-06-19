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
original_dir = os.getcwd()
os.chdir(target_folder)
print("Entering target folder", target_folder)


file = 'last_timestep.lammpstrj'
list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file,unscale=True)

list_TSTEP=[0]
list_Pos = list_ATOMS[:,:,2:]
list_Types = list_ATOMS[:,:,1]
Pos = list_ATOMS[-1][:,2:]
Types = list_ATOMS[-1][:,1]
print(list_BOX)

### Number of atoms ###
n_total = len(Pos)
print("Total number of atoms:", n_total)


OH_atoms = np.where(((Types == 3) |(Types == 4)))[0]
Pos_OH = Pos[OH_atoms]

### Atoms on the contour of the surface ###
Pos_contour, contour_indices_global, contours_per_slice = external_surface(Pos, list_BOX, dz=1.0
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
D, Si_count_O, O_count_Si =  compute_hist_neighbors(Pos_contour,Types_contour,cube=30,threshold_type1=2,threshold_type2=2,threshold_H=1.3,rdf_max=5)

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

os.chdir(original_dir)
os.chdir('mol_file')
mol_file = "1-propanamine.sdf"
type_map = {"C": 5, "H": 6, "N": 7}
surface_area_nm2 = compute_surface_method_3(Pos, list_BOX, dz=1.0)*10**(-2)
density_nm2 = 1.0

print(surface_area_nm2)


(Pos_new, Types_new, Lims, n_grafted,
        Pos_contour_new, Types_contour_new,
        new_pos, new_types,
        all_mol_bonds,  bonds_tpl, terminal_H_idx        # new return value
    ) = graft_molecules(
    Pos, Types, list_BOX, Pos_contour, Types_contour,
    surface_area_nm2, mol_file, type_map, density_nm2,
    
    d_surface        = 1.7219,
    d_mol_surface    = 2.0,
    d_mol_mol        = 3.0,
    neighbor_cutoff  = 10.0,
    
    Si_type          = 1,
    Si_coordination  = 3,
   
    use_OH_sites     = True,
    OH_types         = (3, 4),
    OH_bond_cutoff   = 2.0,
    
    use_O_triangle_sites  = False,
    O_triangle_type       = 2,
    O_triangle_radius     = 100.0,
    O_triangle_bond_cutoff= 100.0,
)



print("n_grafted", n_grafted)


Si_contour_indices = np.where(Types_contour == 1)[0]
D, Si_count_O, O_count_Si =  compute_hist_neighbors(Pos_contour,Types_contour,cube=30,threshold_type1=2,threshold_2=2,threshold_H=1.3,rdf_max=5)
insaturated_Si_mask = np.array([count == 3 for count in Si_count_O])
insaturated_Si_contour = Si_contour_indices[insaturated_Si_mask]
Pos_insat_Si = Pos_contour[insaturated_Si_contour]

N_contour_indices = np.where(Types_contour == 7)[0]
Pos_N = Pos_contour[N_contour_indices]


H_contour_indices = np.where(Types_contour == 6)[0]
Pos_H = Pos_contour[H_contour_indices]


##### -------- Add chromophore on the surface -------- #####

all_anchor_pos_stacked, all_anchor_types_stacked = new_pos, new_types


n_atoms_per_anchor = len(all_anchor_pos_stacked) // n_grafted
chrom_file = "chromophore.mol"
chrom_type_map = {"C": 8, "H": 9, "O": 10}
anchor_type_map = {5: "C", 6: "H", 7: "N"}


anchor_bonds_tpl = bonds_tpl
Pos_new, Types_new, placed, chrom_pos_stacked,chrom_types_stacked, updated_anchor_pos, updated_anchor_types, all_mol_bonds = graft_chromophores(
                                                                                    Pos,                     # surface atom positions (N_surf, 3)
                                                                                    Types,                   # surface atom types    (N_surf,)
                                                                                    all_anchor_pos_stacked,  # output of graft_molecules: stacked anchor positions
                                                                                    all_anchor_types_stacked,# output of graft_molecules: stacked anchor types
                                                                                    n_atoms_per_anchor,      # atom count of ONE anchor molecule (after _prepare_mol)
                                                                                    anchor_type_map,         # {type_id --> element} for anchors
                                                                                    chrom_file,              # XYZ file of the chromophore (e.g. pyrenecarboxylic acid)
                                                                                    chrom_type_map,          # {element --> type_id} for chromophore
                                                                                    anchor_bonds_tpl,        # 1-based local bonds from _prepare_mol
                                                                                    d_amide_bond  = 1.34,    # N–C(=O) bond length  [Å]
                                                                                    d_chrom_surf  = 1.0,     # min allowed distance chromophore ↔ surface  [Å]
                                                                                    d_chrom_chrom = 3.5,     # min allowed distance chromophore ↔ chromophore [Å]
                                                                                )





N_contour_indices = np.where(Types_new == 7)[0]
Pos_N = Pos_new[N_contour_indices]

H_contour_indices = np.where(Types_new == 6)[0]
Pos_H = Pos_new[H_contour_indices]

type_colors = {1: [103, 179, 179], 2: [192, 0, 0], 3: "blue", 4: "green"}
plotter = pv.Plotter()

sphere = pv.Sphere(radius = 0.2)
sphere2 = pv.Sphere(radius = 0.5)
mesh_contour = pv.PolyData(Pos_contour_new).glyph(scale = False, geom = sphere)
plotter.add_mesh(mesh_contour, color=[103, 179, 179], opacity = 0.1)
mesh_insat_Si = pv.PolyData(Pos_insat_Si).glyph(scale = False, geom = sphere2)
plotter.add_mesh(mesh_insat_Si, color='blue', opacity = 1.0)
mesh_mol = pv.PolyData(chrom_pos_stacked).glyph(scale = False, geom = sphere2)
plotter.add_mesh(mesh_mol, color=[255, 0, 0], opacity = 1.0)
mesh_N = pv.PolyData(updated_anchor_pos).glyph(scale = False, geom = sphere2)
plotter.add_mesh(mesh_N, color='green', opacity = 1.0)
plotter.show()





