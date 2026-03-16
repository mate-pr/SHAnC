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


type_map = {"Au": 1}
mass_map = {1: 196.967}

convert_xyz_to_data("Au.xyz", type_map=type_map, mass_map=mass_map, metal = True)

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

diameter = D_list[a]
width = Width_list[a]
pitch = Pitch_list[a]
thickness = Thickness_list[a]
int_thick = 0


### Proportional lists ###


# Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota, diameter,pitch,width,thickness,int_thick, 
#                                                                                                             do_clean=True,circling=True,do_rota_transf=False)






Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int = create_syst(rota, diameter,pitch,width,thickness,int_thick, 
                                                                                                            do_clean=False,circling=True,do_rota_transf=False,
                                                                                                            file_duplicate="Au.data",file_output = "au_dupl.data", 
                                                                                                            file_output_cast = "au_int.data", 
                                                                                                            mass_map = mass_map, metal = True)

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

