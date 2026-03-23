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


##### ------ Initialisation of the system ------- #####


### Real values ###
rota = 1.0
D = 244
P = 453
T = 112
W = 226


### Proportional lists ###
Pitch_list = [200]#, 120, 150, 200, 400]
print("P=", Pitch_list)
D_list = [(p/P)*D for p in Pitch_list]
print("D=", D_list)
Thickness_list = [(p/P)*T for p in Pitch_list]
print("T=", Thickness_list)
Width_list = [(p/P)*W for p in Pitch_list]
print("W=", Width_list)

### Create system and write dump file ###
a=0

diameter = D_list[a]
width = Width_list[a]
pitch = Pitch_list[a]
thickness = Thickness_list[a]

Pos_transfo,Types,Lims_tot,Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int, slide_z, mean = create_syst(rota,D_list[a],Pitch_list[a],Width_list[a],Thickness_list[a],int_thick = 15, do_clean=True,circling=True,do_rota_transf=False)


print(f"{type} Number of Si : ",np.sum(Types==1))
D_calculated = (np.max(Pos_transfo[:,0])-np.min(Pos_transfo[:,0]))
print("Diameter of the system : ",(np.max(Pos_transfo[:,0])-np.min(Pos_transfo[:,0])))          

list_BOX,list_ATOMS = read_data("quartz_dupl.data",do_scale=False)
write_dump("Test_system.lammpstrj",[0], [len(list_ATOMS[0])], list_BOX, list_ATOMS)
file = "Test_system.lammpstrj"

file = "Test_system.lammpstrj"
file = "Tests_systems_with_different_pitch/sys_different_p_vers4_procedure3/Helicetest_P150/last_timestep.lammpstrj"
file = "HeliceAsym_0.4_P200/last_timestep.lammpstrj"
#file = "HeliceAsym_0.4_P200/last_timestep.lammpstrj"
### Read the dump file and analyze surface atoms ###
list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file,unscale=True)

list_TSTEP=[0]
list_Pos = list_ATOMS[:,:,2:]
list_Types = list_ATOMS[:,:,1]
Pos = list_ATOMS[-1][:,2:]
Types = list_ATOMS[-1][:,1]
print(list_BOX)





##### -------- Minimal distance to a slice of the structure --------- #####





### Slice in z of the helix ###
def slice_z(Pos, z0, dz): #Remettre à 0
    mask = (Pos[:,2] >= z0 - dz/2) & (Pos[:,2] <= z0 + dz/2)
    return mask


### Creation of the first circle around the whole slice ###
def circle(Pos, z0, dz, center_mode = "mid", shift = False):

    ## Take a slice ##
    mask = slice_z(Pos, z0, dz)
    Pos_slice = Pos[mask]

    if len(Pos_slice) == 0:
        return None, None, None
    xy = Pos_slice[:, :2]

    ## Find center (mean preferred but mid works) ##
    if center_mode == "mean":
        center = np.mean(xy, axis =0)
    elif center_mode == "mid":
        center = (np.max(xy, axis=0) + np.min(xy, axis = 0))/2
    else:
        return "Err, center_mode = 'mid' or 'mean'"
    
    ## Radius of the slice as the maximum distance between the center and an atom ##
    distances = np.linalg.norm(xy - center, axis = 1)
    R = np.max(distances)

    if shift == True:
        center_full = np.mean(Pos[:,:2], axis = 0)
        center = center_full 
    return mask, center, R


### Creation of the circle points (n_points --> how many points per slice) ###
def circle_min_distances(Pos_slice, center, R, n_points=500):
    xy = Pos_slice[:,:2]

    ## Take points positions around the circle ##
    theta = np.linspace(0, 2*np.pi, n_points, endpoint = False)
    circle_x = center[0] + R*np.cos(theta)
    circle_y = center[1] + R*np.sin(theta)
    circle_pts = np.column_stack((circle_x, circle_y))

    min_dist = np.zeros(n_points)
    diff = circle_pts[:,None,:] - xy[None, :,:]
    dist = np.linalg.norm(diff, axis = 2)
    # dist = sd.cdist(circle_pts, xy)
    min_dist = np.min(dist, axis = 1)

    ## Positions with the lowest distances ##
    closest_indices = np.argmin(dist, axis=1)

    return circle_pts, closest_indices, min_dist, theta


### Build of the second circle based on the first circle ###
def second_circle(circle_pts, min_dist, theta):

    idx_max = np.argmax(min_dist)
    theta_max = theta[idx_max]
    center2 = circle_pts[idx_max]
    r2 = min_dist[idx_max]

    return center2, r2


### Mapping of the external surface ###
def external_surface(Pos, list_BOX, dz = 0.5, n_points = 1000): 
    #0.5 seems okay to calculate --> need analysis to know when stabilizes
    z_min, z_max = np.min(Pos[:,:2]), np.max(Pos[:,:2])
    z_min = list_BOX[0][2][0]
    z_max = list_BOX[0][2][1]
    contour_indices_global = []

    z_values = np.arange(0, z_max, dz)
    contour_per_slice = []
    for z0 in z_values:
        mask = slice_z(Pos, z0, dz)
        Pos_slice = Pos[mask]
        
        if len(Pos_slice)==0:
            continue

        mask_slice, center1, R1 = circle(Pos_slice, z0, dz)
        if mask_slice is None:
            continue
        
        circle_pts1, closest_indices1, min_dist1, theta1 = circle_min_distances(Pos_slice, center1, R1, n_points=n_points)
        center2, R2 = second_circle(circle_pts1, min_dist1, theta1)
        circle_pts2, closest_indices2, min_dist2, theta2 = circle_min_distances(Pos_slice, center2, R2, n_points=n_points)

        slice_indices = np.unique(np.concatenate([closest_indices1, closest_indices2]))
        global_indices = np.where(mask)[0][slice_indices]
        contour_indices_global.extend(global_indices)
        contour_per_slice.append((z0, Pos_slice[slice_indices]))
        
    contour_indices_global = np.unique(contour_indices_global)
    Pos_contour = Pos[contour_indices_global]

    return Pos_contour, contour_indices_global, contour_per_slice
    




##### -------- Visualisation of contour -------- #####




## Visualisation of contour points ##

def visualize_contour(Pos, list_BOX, z0, dz = 1.0, n_points=1000):
    mask = slice_z(Pos, z0, dz)
    Pos_slice = Pos[mask]

    _, center1, R1 = circle(Pos_slice, z0, dz)
    circle_pts1, closest_indices1, min_dist1, theta1 = circle_min_distances(Pos_slice, center1, R1, n_points=n_points)
    
    center2, R2 = second_circle(circle_pts1, min_dist1, theta1)
    circle_pts2, closest_indices2, min_dist2, theta2 = circle_min_distances(Pos_slice, center2, R2, n_points=n_points)

    slice_indices = np.concatenate([closest_indices1, closest_indices2])
    contour_pts = Pos_slice[slice_indices]
    ordered_pts, center = order_contour_trigo(contour_pts)
    P = contour_perimeter(ordered_pts)
    
    xy = ordered_pts[:,:2]
    xy_closed = np.vstack([xy, xy[0]])

    plt.scatter(Pos_slice[:,0], Pos_slice[:,1], s = 10)
    plt.scatter(xy_closed[:,0], xy_closed[:,1], s = 10)
    plt.scatter(circle_pts1[:,0], circle_pts1[:,1], s = 10)
    plt.scatter(circle_pts2[:,0], circle_pts2[:,1], s = 10)

    # n = 0
    # for i, (x, y) in enumerate(xy):
    #     if i%100 == 0:
    #         plt.text(x, y, str(i), fontsize = 8)

    # plt.scatter(center1[0], center1[1])
    # plt.scatter(center2[0], center2[1])
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.xlabel("x (angstrom)")
    plt.ylabel("y (angstrom)")
    plt.show()

# visualize_contour(Pos, list_BOX, 10.0, dz = 1.0, n_points=1000)






##### ----------- Calculation of the surface ----------- #####




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



### Approximation of the surface ###
## Van der Waals Radii of Elements Rowland, 1996 (Si is empirical) in nm^2 ##
approx_surface = 4*(np.pi*n_Si_atoms*(0.215**2) + np.pi*n_O_atoms*(0.158**2) + np.pi*n_H_atoms*(0.11**2)) 
print("Approximate surface with Vdw radius in nm^2:", approx_surface)

## Calculation with the sum of the perimeters over dz ##
def order_contour_trigo(Pos_slice_contour):
    xy = Pos_slice_contour[:,:2]
    center = np.mean(xy, axis = 0)
    dx = xy[:,0] - center[0]
    dy = xy[:,1] - center[1]

    angles = np.arctan2(dy, dx)
    order = np.argsort(angles)

    ordered_points = Pos_slice_contour[order]
    
    return ordered_points, center

def contour_perimeter(ordered_points):
    xy = ordered_points[:,:2]

    xy_closed = np.vstack([xy, xy[0]])
    diffs = np.diff(xy_closed, axis = 0)

    distances = np.linalg.norm(diffs, axis = 1)
    return np.sum(distances)

def compute_surface(contour_per_slice, list_BOX, dz = 1.0, n_points = 1000):
    total_surface = 0.0
    for z0, contour_pts in contour_per_slice:
        if len(contour_pts) < 3:
            continue
        ordered_pts, center = order_contour_trigo(contour_pts)
        P = contour_perimeter(ordered_pts)
        total_surface += P*dz
    return total_surface


approx_surface_with_perim = compute_surface(contour_per_slice=contours_per_slice, list_BOX=list_BOX)*10**(-2)
print("Approximate surface calculated with contour in nm^2:", approx_surface_with_perim)

# ### Plot ###
# plotter = pv.Plotter()
# sphere = pv.Sphere(radius=0.4)
# mesh = pv.PolyData(Pos_contour).glyph(scale=False, geom=sphere)
# mesh_Si = pv.PolyData(Pos_contour[Si_atoms_contour]).glyph(scale=False, geom=sphere)
# mesh_structure = pv.PolyData(Pos).glyph(scale=False, geom=sphere)
# plotter.add_mesh(mesh,color='red')
# plotter.add_mesh(mesh_structure,color='lightgrey',opacity=0.1)
# plotter.add_mesh(mesh_Si,color='blue')
# plotter.show()






##### --------- Add N atoms perpandicular to the surface at unsaturated Si sites ------------ #####





Si_N_dist = 1.9 #Angstrom
min_clearance = 1.0 #min disatance from any other atom
neighbor_cutoff = 10.0 #radius to estimate the direction to put N (opposite to bulk direction)
tree_all = cKDTree(Pos) #tree over all atoms to detect collision and bulk direction


Si_contour_indices = np.where(Types_contour == 1)[0]
D, Si_count_O, O_count_Si =  compute_hist_neighbors(Pos_contour,Types_contour,cube=30,threshold_Si=2,threshold_O=2,threshold_H=1.3,rdf_max=5)
insaturated_Si_mask = np.array([count == 3 for count in Si_count_O])
insaturated_Si_contour = Si_contour_indices[insaturated_Si_mask]
Pos_insat_Si = Pos_contour[insaturated_Si_contour]


def compute_outward_normal_density(Si_pos, tree, Pos_all, cutoff = 6.0):
    neighbor_ids = tree.query_ball_point(Si_pos, r=cutoff)
    if len(neighbor_ids) < 2:
        return np.array([0.0, 0.0, 1.0])
    neighbor_pos = Pos_all[neighbor_ids]

    inward = np.mean(neighbor_pos - Si_pos, axis = 0)
    inward_norm = np.linalg.norm(inward)

    if inward_norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    
    outward = -inward/inward_norm
    return outward

print(f"Adding N atoms at {len(Pos_insat_Si)} insaturated Si sites")

N_positions = []
skipped = 0
for Si_pos in Pos_insat_Si:
    outward = compute_outward_normal_density(Si_pos, tree_all, Pos, cutoff=neighbor_cutoff)
    N_candidate = Si_pos + Si_N_dist*outward

    nearby = tree_all.query_ball_point(N_candidate, min_clearance)
    if len(nearby) > 0:
        skipped +=1
        continue
    
    N_positions.append(N_candidate)

N_positions = np.array(N_positions) if N_positions else np.empty((0,3))
N_types = np.full(len(N_positions), 8)
print("N_positions skipped:",skipped)

Pos_new = np.vstack([Pos, N_positions])
Types_new = np.concatenate([Types, N_types])

Pos_contour_new = np.vstack([Pos_contour, N_positions])
Types_contour_new = np.concatenate([Types_contour, N_types])

print("New total atoms:",len(Pos_new))
print("New total contour atoms:", len(Pos_contour_new))

plotter = pv.Plotter()
sphere = pv.Sphere(radius = 0.4)
sphere2 = pv.Sphere(radius = 0.8)
mesh_contour = pv.PolyData(Pos_contour_new).glyph(scale = False, geom = sphere)
plotter.add_mesh(mesh_contour, color='lightgrey', opacity = 0.1)
mesh_insat_Si = pv.PolyData(Pos_insat_Si).glyph(scale = False, geom = sphere2)
plotter.add_mesh(mesh_insat_Si, color='blue', opacity = 1.0)
mesh_N = pv.PolyData(N_positions).glyph(scale = False, geom = sphere2)
plotter.add_mesh(mesh_N, color='red', opacity = 1.0)
plotter.show()
