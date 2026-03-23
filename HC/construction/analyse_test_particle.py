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
from scipy.spatial import cKDTree
import scipy.spatial.distance as sd
import numpy as np
from sys_surf import external_surface, compute_surface




file = "test_particle/test_particle_vers2/13085092/last_timestep.lammpstrj"
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

### Atoms on the contour of the surface ###
Pos_contour, contour_indices_global, contours_per_slice = external_surface(Pos, list_BOX)

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
D, Si_count_O, O_count_Si =  compute_hist_neighbors(Pos_contour,Types_contour,cube=30,threshold_Si=2,threshold_O=2,threshold_H=1.3,rdf_max=5)

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

Si_single_tot = np.where(Types == 5)[0]
Si_single = np.where(Types_contour == 5)[0]
print("Total number of Si single atoms on the surface:", len(Si_single), "over total number of Si_single:", len(Si_single_tot) )






#### Analysis of the surface from test_particle ####

### Atom types ###

Type_labels = ["", "Si", "O", "Oh", "H", "Si", "Oh", "H"]
Type_thresholds = np.array([0.0, 2.0, 2.0, 2.0 ,2.0, 2.0, 2.0, 2.0])
Type_expected_bond = np.array([0, 4, 2, 2, 1, 4, 2, 1])


N_Types = len(Type_labels)
Anchor_type = 5
Anchor_bonds = 3
rdf_max = 5.0
cube = 100


def compute_bonds3(Pos, Types, source_type, target_types, is_type_5_O = False):
    Pos_src = Pos[Types == source_type]
    counts = np.zeros((len(Pos_src)))
    for ttype in target_types:
        if is_type_5_O:
            thresh = min(Type_thresholds[3], 
                     Type_thresholds[ttype])
        else: 
            thresh = min(Type_thresholds[source_type], 
                     Type_thresholds[ttype])
        D = sd.cdist(Pos_src, Pos[Types == ttype])
        counts += (D < thresh).sum(axis = 1)
    
    return counts

def saturation_check(type_idx, bond_count):
    expected = Type_expected_bond[type_idx]
    label = Type_labels[type_idx]
    total = len(bond_count)
    saturated = int((bond_count == expected).sum())
    unsaturated = total - saturated
    unique, counts = np.unique(bond_count, return_counts=True)
    bond_distribution = list(zip(unique, counts))
    print(label, "total", total)
    print("unsaturated", unsaturated)
    print("bond distribution", bond_distribution)
    return saturated, unsaturated

print("total atoms", len(Pos))
Pos_contour, contour_indices_global, contour_per_slice = external_surface(Pos, list_BOX)
print("surface atoms", len(Pos_contour))

is_type_5_O = False
present_types = np.unique(Types_contour)
if is_type_5_O:
    Is_O = np.array([False, False, True, True, False, True, False, False])
    o_like_types = present_types[Is_O[present_types].astype('bool')]
    Is_Si = np.array([False, True, False, False, False, False, False, False])
    si_types = present_types[Is_Si[present_types].astype('bool')]
else:
    Is_O = np.array([False, False, True, True, False, False, True, False])
    o_like_types = present_types[Is_O[present_types].astype('bool')]
    Is_Si = np.array([False, True, False, False, False, True, False, False])
    si_types = present_types[Is_Si[present_types].astype('bool')]

Is_H = np.array([False, False, False, True, False, False, False, True])
h_types = o_like_types

bond_counts_by_type = [None]*N_Types
for t in present_types:
    if Is_Si[t]:
        bound_to = o_like_types
    elif Is_O[t]:
        bound_to = si_types
    elif Is_H[t]:
        bound_to = h_types
    else:
        continue
    bc = compute_bonds3(Pos_contour, Types_contour, t, bound_to)
    bond_counts_by_type[t] = bc
    saturation_check(t, bc)

cube = 100

Lx, Ly, Lz = np.max(Pos, axis = 0)
lx, ly, lz = np.min(Pos, axis = 0)

Nx = int((Lx - lx )//cube + 1)
Ny = int((Ly - ly )//cube + 1)
Nz = int((Lz - lz )//cube + 1)

Num_at = len(Types_contour)
pad_top = Pos_contour[:,2] > (Lz - rdf_max)
pad_bot = Pos_contour[:,2] > (lz + rdf_max)

Pos_added = np.vstack([Pos_contour, Pos_contour[pad_top] - [0, 0, Lz - lz],
                       Pos_contour[pad_top] + [0, 0, Lz - lz]])
Types_added = np.vstack([Types_contour, Types_contour[pad_top], Types_contour[pad_bot]])
In_trunc = np.zeros(len(Pos_added), dtype = bool)
In_trunc[:Num_at] = True

if is_type_5_O:
    Anchor_type = 1
    anchor_global_indices = np.where(Types_contour == Anchor_type)
else:
    anchor_global_indices = np.where(Types_contour == Anchor_type)
    Num_anchor = len(anchor_global_indices)
    anchor_bond_counts = np.zeros(Num_anchor, dtype = int)
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                mask_unique = (
                    (Pos_added[:,0] >= x*cube +lx)&
                    (Pos_added[:,0] < (x+1)*cube +lx)&
                    (Pos_added[:,1] >= y*cube +ly)&
                    (Pos_added[:,1] < (y+1)*cube +ly)&
                    (Pos_added[:,2] >= z*cube +lz)&
                    (Pos_added[:,2] < (z+1)*cube +lz)
                )
                mask_ext = (
                    (Pos_added[:,0] >= x*cube +lx -rdf_max)&
                    (Pos_added[:,0] < (x+1)*cube +lx +rdf_max)&
                    (Pos_added[:,1] >= y*cube +ly -rdf_max)&
                    (Pos_added[:,1] < (y+1)*cube +ly +rdf_max)&
                    (Pos_added[:,2] >= z*cube +lz -rdf_max)&
                    (Pos_added[:,2] < (z+1)*cube +lz +rdf_max)
                )
                Pos_ext = Pos_added[mask_ext]
                Types_ext=Types_added[mask_ext]

                if not (Types_ext == Anchor_type).any():
                    continue
                counts = compute_bonds3(Pos_ext, Types_ext, Anchor_type, o_like_types)

                keep = (mask_unique & In_trunc)[mask_ext][Types_ext == Anchor_type]
                counts_to_store = counts[keep]
                cell_anchor_pos = Pos_ext[Types_ext == Anchor_type][keep]
                dists = sd.cdist(cell_anchor_pos, Pos_contour[anchor_global_indices])
                ranks = np.argmni(dists, axis = 1)
                anchor_bond_counts[ranks] = counts_to_store
    

unique_b, counts_b = np.unique(anchor_bond_counts, return_counts=True)
print("Bond_distribution", list(zip(unique_b, counts_b)))
n_anchor_points = int((anchor_bond_counts == Anchor_bonds).sum())
n_oversaturated = int((anchor_bond_counts > Anchor_bonds).sum())
n_unsaturated = int((anchor_bond_counts < Anchor_bonds).sum())

print("Number of anchor points", n_anchor_points)
print("Number of oversaturated points", n_oversaturated)
print("Number of undersaturated anchor points", n_unsaturated)
























# def compute_bonds2(Pos,Types,threshold_Si=2,threshold_O=2,threshold_H=1.3,do_count_type_3=True, do_count_type_5 = True):
#     """
#     compute_bonds(Pos,Types,threshold_Si=2,threshold_O=2,threshold_H=1.3,do_count_type_3=True)

#     Compute the Bonds of a silica system.
#     |!| This version must only be used for small systems. It takes too much memory otherwise.
#     The types are the following : 1 : Si, 2: O, 3: Oh, 4: H

#     Parameters
#     ----------
#     Pos : array
#         The position of the atoms
#     Types : array
#         The type of the Atoms
#     threshold_Si : float, optional
#         The threshold used to consider if Si and O are bonding. 2 by default
#     threshold_O : float, optional
#         The threshold used to consider if O and Si are bonding. 2 by default
#     threshold_H : float, optional
#         The threshold used to consider if O and H are bonding. 1.3 by default
#     do_count_type_3 : bool, optional
#         If one wants to consider the Oh in the calculations.

#     Returns
#     -------
#         Bonds, Si_count_O, O_count_Si, O_count_H, H_count_O
#     """

#     Pos_Si_single = Pos[Types==5]
#     Pos_O = Pos[((Types==2)).astype("bool")]
   
  

#     # There are multiple ways to compute the distance, but this one is the fastest I found
#     Dist = sd.cdist(Pos_Si_single,Pos_O)

#     Bonds_Si_single = (Dist<(threshold_Si))
#     Bonds_O = (Dist<(threshold_O))
#     Bonds =  Bonds_Si_single + Bonds_O


#     Si_single_count_O = np.sum(Bonds,axis=1)
   

#     return Bonds, Si_single_count_O




# Lx,Ly,Lz = np.max(Pos,axis=0)
# lx,ly,lz = np.min(Pos,axis=0)


# cube = 100

# Nx = int((Lx - lx) // cube + 1)
# Ny = int((Ly - ly) // cube + 1)
# Nz = int((Lz - lz) // cube + 1)

# Pos_added = np.copy(Pos_contour)
# Types_added = np.copy(Types_contour)

# Num_Si_or = np.sum(Types_contour==1)

# rdf_max = 5
# threshold_Si=4
# threshold_O=2
# threshold_H=1

# Pos_add_z = Pos_contour[:,2] > (Lz - 5)
# Pos_remove_z = Pos_contour[:,2] < (lz + 5)

# Pos_add_Lz = Pos_contour[Pos_add_z] - np.array([[0,0,Lz-lz]])
# Pos_remove_Lz = Pos_contour[Pos_remove_z] + np.array([[0,0,Lz-lz]])

# Pos_add = np.append(Pos_add_Lz,Pos_remove_Lz,axis=0)
# Pos_added = np.append(Pos_added,Pos_add,axis=0)

# Types_add = np.append(Types_contour[Pos_add_z],Types_contour[Pos_remove_z],axis=0)
# Types_added = np.append(Types_added,Types_add,axis=0)


# Num_at = len(Types_contour)

# Num_Si_single = np.sum(Types_contour==5)
# In_trunc = np.array([1]*Num_at + [0]*(len(Pos_added)-Num_at),dtype="bool")


# Si_single_count_O_tot = np.zeros((Num_Si_single))
# Dist_list = []
# for x in range(Nx):
#     for y in range(Ny):
#         for z in range(Nz):
#             #the first slicing is for the system that will the computation will be done to
#             Pos_trunc_x_u = (Pos_added[:,0] >= (x*cube + lx)) * (Pos_added[:,0] < ((x+1)*cube + lx))
#             Pos_trunc_y_u = (Pos_added[:,1] >= (y*cube + ly)) * (Pos_added[:,1] < ((y+1)*cube + ly))
#             Pos_trunc_z_u = (Pos_added[:,2] >= (z*cube + lz)) * (Pos_added[:,2] < ((z+1)*cube + lz))
#             Ind_trunc_uniq = Pos_trunc_x_u * Pos_trunc_y_u * Pos_trunc_z_u
#             Pos_trunc_uniq = Pos_added[Ind_trunc_uniq]
#             Types_trunc_uniq = Types_added[Ind_trunc_uniq]

#             #The second slicing is for the RDF, as it needs a larger distance
#             Pos_trunc_x = (Pos_added[:,0] >= (x*cube + lx - rdf_max)) * (Pos_added[:,0] < ((x+1)*cube + lx + rdf_max))
#             Pos_trunc_y = (Pos_added[:,1] >= (y*cube + ly - rdf_max)) * (Pos_added[:,1] < ((y+1)*cube + ly + rdf_max))
#             Pos_trunc_z = (Pos_added[:,2] >= (z*cube + lz - rdf_max)) * (Pos_added[:,2] < ((z+1)*cube + lz + rdf_max))
#             Pos_trunc_ind = Pos_trunc_x * Pos_trunc_y * Pos_trunc_z
#             Pos_trunc = Pos_added[Pos_trunc_ind]
#             Types_trunc = Types_added[Pos_trunc_ind]

#             Ind_trunc_uniq_in_trunc = (Ind_trunc_uniq * In_trunc)[Pos_trunc_ind]

#             if (Types_trunc == 5).any():

#                 Si_single_count_O = compute_bonds2(Pos_trunc,Types_trunc,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H,do_count_type_3=True)[1]
#                 Si_single_count_O = Si_single_count_O[Ind_trunc_uniq_in_trunc[(Types_trunc==5).astype("bool")]]
#                 Si_index = Ind_trunc_uniq[:Num_at][((Types_contour==5)).astype("bool")]
#                 Si_single_count_O_tot[Si_index] = Si_single_count_O

#                 Pos_Si_single = Pos_trunc_uniq[Types_trunc_uniq==5]


# Si_single_count_O_tot = Si_single_count_O_tot[:Num_Si_single]

# print(Si_single_count_O_tot)

# n_Si_single_atoms_insat = 0
# n_Si_single_good_insat = 0
# for Si_single in Si_single_count_O_tot:
#     if (Si_single == 2) or (Si_single == 1):
#         n_Si_single_atoms_insat +=1
#     if (Si_single == 3):
#         n_Si_single_good_insat +=1

# print("Si single atoms insaturated:",n_Si_single_atoms_insat, "while good insaturation Si-O3:", n_Si_single_good_insat)