import distorsion
import matplotlib.pyplot as plt
import pyvista as pv 
import script_analysis
import read_write
import os 
from read_write import *
from script_analysis import *
from distorsion import *
from pathlib import Path
from analysis import *







# import os
# import numpy as np
# import argparse

# # Parsing des arguments
# parser = argparse.ArgumentParser(description="Convert POSCAR or CONTCAR file to XYZ format.")
# # parser.add_argument("files", type=str, help="Path to the POSCAR or CONTCAR file")
# # args = parser.parse_args()
# # files = args.files
# file = "files_lattices/TiO2-rutile.poscar"
# k, l = 0, 0
# atlist = []

# try:
#     print('try')
#     with open(file, 'r') as file:
#         for _ in range(2):
#             file.readline()
#         xa, ya, za = map(float, file.readline().split())
#         xb, yb, zb = map(float, file.readline().split())
#         xc, yc, zc = map(float, file.readline().split())
#         attype = file.readline().split()
#         atnum = file.readline().split()
#         selective_dynamics = False
#         line = file.readline().strip()
#         contains_selective = "selective" in line.lower()
#         if contains_selective:
#             file.readline()
#         for i in atnum:
#             for j in range(int(i)):
#                 write = " ".join(file.readline().split()[:3])
#                 atlist.append(f"{attype[k]} {write}")
#             k += 1
# except Exception as e:
#     print(f"Error while reading the file: {e}")
#     exit(1)

# cell_matrix = np.array([[xa, xb, xc], [ya, yb, yc], [za, zb, zc]])

# try:
#     print('try')
#     for line in atlist:
#         with open(f'files_lattices/TiO2-rutile-2.xyz', 'w') as outfile:
#             l = sum(int(i) for i in atnum)
#             outfile.write(f"{l}\n")
#             outfile.write("Generated from POSCAR\n")
#             for line in atlist:
#                 at, x, y, z = line.split()
#                 coords = list(map(float, (x, y, z)))
#                 x, y, z = np.dot(cell_matrix, coords)
#                 outfile.write(f"{at} {x:.16f} {y:.16f} {z:.16f}\n")
# except Exception as e:
#     print(f"Error while writing the file: {e}")
#     exit(1)




# import spglib
# import numpy as np

# # Cell parameters (from LAMMPS box)
# a, b, c = 5.08470, 5.08470, 7.09858

# # Lattice vectors (rows)
# lattice = np.array([
#     [a, 0, 0],
#     [0, b, 0],
#     [0, 0, c]
# ])

# # Cartesian positions → fractional
# positions_cart = np.array([
#     [3.587511, 3.587511, 6.561636],  # Si
#     [4.039541, 1.04516,  1.237701],  # Si
#     [1.04516,  4.039541, 4.786992],  # Si
#     [1.49719,  1.49719,  3.012347],  # Si
#     [4.606215, 3.859176, 5.323935],  # O
#     [0.478486, 1.225525, 1.774645],  # O
#     [3.859176, 4.606215, 0.700758],  # O
#     [1.316826, 3.020836, 3.549291],  # O
#     [1.225525, 0.478486, 4.250048],  # O
#     [2.063865, 3.767875, 6.024693],  # O
#     [3.020836, 1.316826, 2.475402],  # O
#     [3.767875, 2.063865, 0.0      ],  # O
# ])

# positions_frac = positions_cart / np.array([a, b, c])

# # Atom types: 1=Si, 2=O  →  spglib just needs integers
# numbers = [14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8]

# cell = (lattice, positions_frac, numbers)

# # --- Analysis ---
# sym = spglib.get_symmetry_dataset(cell, symprec=1e-3)

# print(f"Space group:     {sym['international']}  (No. {sym['number']})")
# print(f"Hall symbol:     {sym['hall']}")
# print(f"Crystal system:  {spglib.get_spacegroup(cell, symprec=1e-3)}")
# print(f"Wyckoff letters: {sym['wyckoffs']}")
# print(f"Equivalent atoms:{sym['equivalent_atoms']}")


# type_map = {"Si": 1, "O": 2}
# mass_map = {1: 28.0855, 2: 15.9994}


type_map = {"Zr": 1, "O": 2}
mass_map = {1: 91.224, 2: 15.9994}

# type_map = {"Ag": 1}
# mass_map = {1:107.8682}


# convert_xyz_to_data("files_lattices/ZrO2-rutile.xyz", type_map=type_map, mass_map=mass_map)
# # convert_cif_to_data("files_lattices/SiO2.cif", type_map=type_map, mass_map=mass_map)


### Real values for Si ###
rota = 1.0
D = 244
P = 453
T = 112
W = 226


### Proportional lists ###
Pitch_list = [100, 120, 150, 200, 400]
print("P=", Pitch_list)
D_list = [(p/P)*D for p in Pitch_list]
print("D=", D_list)
Thickness_list = [(p/P)*T + 10 if p< 200 else (p/P)*T for p in Pitch_list]
print("T=", Thickness_list)
Width_list = [(p/P)*W for p in Pitch_list]
print("W=", Width_list)
Int_thick_list = [10, 10, 10, 15, 15]

a = 2

### 
# diameter = 200
# width = 100
# pitch = 100
# thickness = 30
# int_thick = 0

###
diameter = D_list[a]
width = Width_list[a]
pitch = Pitch_list[a]
thickness = Thickness_list[a]
int_thick = 15

# ### Proportional lists ###

Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota, diameter,pitch,width,thickness,int_thick, 
                                                                                                            do_clean=True,circling=True,do_rota_transf=False, 
                                                                                                            file_duplicate="ZrO2-rutile.data",file_output = "zrO2_dupl.data", 
                                                                                                            file_output_cast = "zrO2_int.data", 
                                                                                                            mass_map = mass_map)






# Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota, diameter,pitch,width,thickness,int_thick, 
#                                                                                                             do_clean=False,circling=True,do_rota_transf=False,
#                                                                                                             file_duplicate="Au.data",file_output = "au_dupl.data", 
#                                                                                                             file_output_cast = "au_int.data", 
#                                                                                                             mass_map = mass_map, metal = True)

# convert_data_to_xyz("Au_dupl.data")

# plt.scatter(Pos_transfo[:,0], Pos_transfo[:,1], s = 10)
# plt.show()
# print(f"{type} Number of Au : ",np.sum(Types==1))
# D_calculated = (np.max(Pos_transfo[:,0])-np.min(Pos_transfo[:,0]))
# print("Diameter of the system : ",(np.max(Pos_transfo[:,0])-np.min(Pos_transfo[:,0])))          

# list_BOX,list_ATOMS = read_data("au_int.data",do_scale=False)

# list_TSTEP=[0]
# list_Pos = list_ATOMS[:,:,2:]
# list_Types = list_ATOMS[:,:,1]
# Pos = list_ATOMS[-1][:,2:]
# Types = list_ATOMS[-1][:,1]

# plotter = pv.Plotter()
 
# mesh = pv.PolyData(Pos.astype(float))
# plotter.add_mesh(mesh)
# plotter.show()
# analyze_mult(list_TSTEP,list_Pos,list_Types,periodic=False,Lims=list_BOX[-1],save=False)

# convert_data_to_xyz("Au_dupl.data")
# convert_data_to_xyz("Au_int.data")

