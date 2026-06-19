import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Construction.import_libraries import *
from Construction.distortion import *
from Construction.analysis import *
from Construction.read_write import *

### SiO2 ###
target_folder = 'sio2_pitch_200'

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

min_x,max_x = np.min(Pos[:,0]),np.max(Pos[:,0])
min_y,max_y = np.min(Pos[:,1]),np.max(Pos[:,1])
min_z,max_z = np.min(Pos[:,2]),np.max(Pos[:,2])

print("Diameter along x: ", max_x - min_x)
print("Diameter along y: ", max_y - min_y)

print("Diameter of the system (mean):",((max_x-min_x) + (max_y-min_y))/2)

print("Pitch of the system after relaxation:", max_z - min_z)

# Plots and visualisation of the structure
analyze_mult(list_TSTEP,list_Pos,list_Types,periodic=False,Lims=Lims,save = True) # Plot rdf, bond analysis
analyze_defects(Pos,Types,periodic=True,Lims=Lims)
analyze_density(Pos)
os.chdir(original_dir)


### Gold ###

target_folder = 'au_pitch_200/au_helix'

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

min_x,max_x = np.min(Pos[:,0]),np.max(Pos[:,0])
min_y,max_y = np.min(Pos[:,1]),np.max(Pos[:,1])
min_z,max_z = np.min(Pos[:,2]),np.max(Pos[:,2])

print("Diameter along x: ", max_x - min_x)
print("Diameter along y: ", max_y - min_y)

print("Diameter of the system (mean):",((max_x-min_x) + (max_y-min_y))/2)

print("Pitch of the system after relaxation:", max_z - min_z)

# Plots and visualisation of the structure
plot_rdf_metal(Pos, Types, vline=2.88, font_weight='normal', title_font_weight='normal')
check_metal_structure(Pos, Types)
analyze_density(Pos)
