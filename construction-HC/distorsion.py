import numpy as np
from script_analysis import *
from read_write import *
from scipy.spatial import KDTree




##### ------ Duplicate a lattice ------- ######
def duplicate(Nx_list,Ny_list,Nz,Lims,Atom_type,Atom_pos,Bonds_OH=[],Angles_OH=[]):
    """
    duplicate(Nx_list,Ny_list,Nz,Lims,Atom_type,Atom_pos,clean=False,Bonds_OH=[],Angles_OH=[])

    Duplicates a system multiple times.
    Nx_list and Ny_list can be a list or an int.
    Int are used to duplicate a certain amount of time
    Lists are used to create hollow structures. Typically [0,3,17,20] will duplicate for the positions 0,1,2, 17,18,19

    The data OH are used to also duplicate the bonds and angles as the H are added using harmonic potential

    Parameters
    ----------
        Nx_list : list or int,
            See description
        Ny_list : list or int,
            See description
        Nz : int,
            The amount of times the initial structure is duplicated
        Lims : list,
            The limit coordinates of the system
        Atom_type : list,
            The type of the atoms
        Atom_pos : list,
            the position of the atoms
        Bonds_OH : list, optional
            The indices of the bonds OH
        Anlges_OH : list, optional
            The indices of the angles SiOH

    Returns
    -------
        Pos : The updated positions
        Types : The updated Types
        Lims_tot : The updated limits
        Bonds_OH_tot : The updated bonds
        Angles_OH_tot : The updated angles

    """
    lx,Lx = Lims[0]
    ly,Ly = Lims[1]
    lz,Lz = Lims[2]

    Nx_0,Ny_0 = 0,0

    bonds = False
    if len(Bonds_OH) > 0:
        bonds = True

    Pos = []
    Types = []
    Pos_O = Atom_pos[Atom_type==2]
    num_O = np.arange(len(Atom_pos))[Atom_type==2]
    z_prop_add = 0

    #Check if int isntead of list
    flag_x = 0
    flag_y = 0
    if type(Nx_list) is int:
        Nx = Nx_list
        flag_x = 1

    elif len(Nx_list) == 2:
        Nx_0 = Nx_list[0]
        Nx = Nx_list[1]
        flag_x = 1

    else: Nx = Nx_list[-1]

    if type(Ny_list) is int:
        Ny = Ny_list
        flag_y = 1

    elif len(Ny_list) == 2:
        Ny_0 = Ny_list[0]
        Ny = Ny_list[1]
        flag_y = 1

    else: Ny = Ny_list[-1]

    for x in range(Nx_0,Nx):
        for y in range(Ny_0,Ny):
            #Check if inside or surface
            if flag_x or flag_y or (x >= Nx_list[0] and x< Nx_list[1]) or (x >= Nx_list[2] and x <= Nx_list[3]) or (y >= Ny_list[0] and y < Ny_list[1]) or (y >= Ny_list[2] and y <= Ny_list[3]):
                for z in range(Nz):
                    for at,ty,num in zip(Atom_pos,Atom_type,range(len(Atom_pos))):
                        Pos.append(at + np.array([x*Lx+lx,y*Ly+ly,z*Lz+lz]))
                        Types.append(ty)

    Pos = np.array(Pos).reshape(len(Pos),3)
    min_x,max_x = np.min(Pos[:,0]),np.max(Pos[:,0])
    min_y,max_y = np.min(Pos[:,1]),np.max(Pos[:,1])
    dx, dy  = (max_x), (max_y)
    max_d = max(dx,dy)
    maxx =  max_d * (2**(1/2))
    maxy =  max_d * (2**(1/2))

    Lims_tot = np.array([[-maxx,maxx],[-maxy,maxy],[lz,Lz]])

    num_at = len(Atom_pos)
    Bonds_OH_tot = []
    Angles_OH_tot = []
    if bonds:
        for j in range(Nz):
            for bond in Bonds_OH:
                Bonds_OH_tot.append([bond[0]+num_at*j,bond[1]+num_at*j])
            for angle in Angles_OH:
                Angles_OH_tot.append([angle[0]+num_at*j,angle[1]+num_at*j,angle[2]+num_at*j])
        return Pos,np.array(Types),Lims_tot, Bonds_OH_tot, Angles_OH_tot

    else:
        return Pos,np.array(Types),Lims_tot, [], []









##### ------ For oxydes : Cleaning the structure by adding H2O ------- #####
def clean_structure(Pos,Types,Lims,N,periodic=True):
    """
    clean_structure(Pos,Types,Lims,N,periodic=True)

    Cleans the structure by adding H2O in the system.
    This is used inside create_syst on the slab.
    This clean will add bonds and angles to the system for lammps to use

    Please note that the added OH will be done so on the x axis and suppose an orthogonal system

    Parameters
    ----------
        Pos : list,
            Position of the atoms
        Types : list,
            Type of the atoms
        Lims : list,
            the limit of the non duplicated system
        N : list,
            the list of the three number of duplication
        periodic : bool, optional
            Computes the bonds as a periodic system. By default True

    Returns
    -------
        New_Pos : The new positions
        New_Types : The new types
        Bonds_OH_corrected : The Bonds of OH
        Angles_OH : The angles SiOH
    """


    Center = [(Lims[0][1]*N[0][0] - Lims[0][0])/2,(Lims[1][1]*N[1][0] - Lims[1][0])/2, (Lims[2][1]*N[2][0] - Lims[2][0])/2]
    Dist_list, Si_count_O, O_count_Si = compute_hist_neighbors(Pos,Types,Lims=Lims*N,periodic=periodic)[:3]

    Pos_Si = Pos[Types==1]
    Pos_O = Pos[Types==2]

    #The Si that have only 2 bonds are located on the edges
    Lack_Si = Pos_Si[Si_count_O == 2]

    Atoms_add_pos = []
    Atoms_add_types = []
    Bonds_OH = []
    num_at = len(Pos)

    for Si in Lack_Si:
        #Add OH next to a Lack_Si on the x axis at 1.6 A and 2.6 A
        Atom_O_add = np.sign(Si[0]-Center[0]) * np.array([1.6,0,0]) + Si
        Atom_H_add = np.sign(Si[0]-Center[0]) * np.array([2.6,0,0]) + Si

        Atoms_add_pos.append(Atom_O_add)
        Atoms_add_pos.append(Atom_H_add)
        Atoms_add_types.append(3)
        Atoms_add_types.append(4)
        Bonds_OH.append([num_at,num_at+1])


        #Get the closest O symmetric with respect to the plane yz and xz
        Pos_O_trunc = Pos_O[(Pos_O[:,2] > (Si[2]-2)) * (Pos_O[:,2] < (Si[2]+2))]
        #The factor should be 2, but 2.2 is used in order to make sure to not put the OH inside
        Si_symm = np.array([2*Center[0] -Si[0],Si[1], Si[2]])
        #This suppose a orthogonal system
        Dist_O = np.sum((Pos_O_trunc - Si_symm.reshape((1,3)))**2,axis=1)
        Pos_O_add_H = Pos_O_trunc[np.argmin(Dist_O)]
        #Add the H atom to the symmetric O
        Atom_H_add = np.sign(Pos_O_add_H-Center[0]) * np.array([1.0,0,0.0]) + Pos_O_add_H
        Atoms_add_pos.append(Atom_H_add)
        Atoms_add_types.append(4)

        #Transform the O to the proper type
        Index_O_trunc = (Types==2) * (Pos[:,2] > (Si[2]-2)) * (Pos[:,2] < (Si[2]+2))
        Index_O_bond = np.argmax((np.cumsum(Index_O_trunc)-1)==np.argmin(Dist_O))
        Types[Index_O_bond] = 3

        Bonds_OH.append([Index_O_bond,num_at+2])

        num_at += 3


    #Appends everyting together
    New_Pos, New_Types = np.append(Pos,Atoms_add_pos,axis=0), np.append(Types,Atoms_add_types,axis=0)
    Middle = (New_Pos[:,2] >= (Lims[2,1]+Lims[2,0])) * (New_Pos[:,2] < (Lims[2,1]*2+Lims[2,0]))
    New_Pos = New_Pos[Middle]
    New_Pos = New_Pos - np.array([0,0,np.min(New_Pos[:,2])])
    New_Types = New_Types[Middle]

    Bonds_OH_corrected = []
    New_index = (np.cumsum(Middle)-1) * Middle


    for bond in Bonds_OH:
        if New_index[bond[0]] != 0 and New_index[bond[1]] != 0:
            Bonds_OH_corrected.append([New_index[bond[0]]+1,New_index[bond[1]]+1])

    #Bonds : O then H
    Angles_OH = []
    Pos_Si = New_Pos[New_Types==1]


    #Compute the angles using the closest Si to the OH
    for bond in Bonds_OH_corrected:
        O = New_Pos[bond[0]-1]
        Pos_Si_trunc =  Pos_Si[(Pos_Si[:,2] > (O[2]-2)) * (Pos_Si[:,2] < (O[2]+2))]
        Dist_Si = np.sum((Pos_Si_trunc - O.reshape((1,3)))**2,axis=1)

        Index_Si_trunc = (New_Types==1) * (New_Pos[:,2] > (O[2]-2)) * (New_Pos[:,2] < (O[2]+2))
        Index_Si_bond = np.argmax((np.cumsum(Index_Si_trunc)-1) == (np.argmin(Dist_Si)))
        Angles_OH.append([Index_Si_bond+1,bond[0],bond[1]])

    return New_Pos, New_Types, Bonds_OH_corrected, Angles_OH









def prepare_slab_with_padding(Atom_pos, Atom_types, Lims, Nx, Ny, Nz, pad = 1):
    lx = Lims[0][1] - Lims[0][0]
    ly = Lims[1][1] - Lims[1][0]
    Pos_pad, Types_pad, Lims_pad, _, _ = duplicate(Nx + 2*pad, Ny + 2*pad, Nz, Lims, Atom_types, Atom_pos)
    ix = (Pos_pad[:,0] - Lims[0][0])//lx
    iy = (Pos_pad[:,1] - Lims[1][0])//ly

    mask_inner = ((ix >= pad) & (ix < Nx+pad)&
                  (iy >= pad) & (iy < Ny+pad))
    
    return Pos_pad, Types_pad, Lims_pad, mask_inner








##### ------ Transformation of the cuboid to get the helix ------ #####
def transfo(Pos,Types,Lims,mass_map, metal = False, slide_z=0, mean = None,D=0,rota=0,enlarge=10,enlarge_z=0.700758, do_periodic=True,other_mapping=False,params_helix=[]):
    """
    transfo(Pos,slide_z=0,D=0,rota=0,enlarge=10,enlarge_z=0.700758,do_periodic=True,circling=True,do_rota_transf=False)

    Do the coordinate transformations of the initial cuboid.
    The formula used for the circling is the elliptical grid mapping


    Parameters
    ----------
        Pos : List,
            The transformed position of the atoms
        slide_z : float, optional
            The amount of shifting in the z coordinate. If 0, will automatically put the lowest atom to 0
            This is used to slide the transformed cast inside the helix
            By default : 0
        rota : float, optional
            The amount of turn to do. By default 0
        enlarge : float, optional
            The space in Angstrom that is added in x and y. This is used to create empty space around the helix. By default 10
        enlarge_z : float, optional
            The space in A that is added in z. The value represent the distance between the highest atom and the border.
            By default 0.700758
        do_periodic : bool, optional
            do the creation of a periodic system after the transformation
        circling : bool, optional
            Circle the basis of the cuboid. This should be used to create helices as is gives much better curvatures. By default True
        do_rota_transf : bool, optional,
            Use the rota transformation instead of the helix one. This should not be used as it gives much worse structure. By default False

    Returns
    -------
        Pos_transfo : The transformed positions
        Lims : the limits of the new system
        slide_z : the amount of shift in the z coordinate the system has taken through the transformation
    """

    mean = np.mean(Pos,axis=0)
    mean[2] = np.min(Pos[:,2])
    Pos = Pos - mean
    print("visualize realign the center")

    Lz = np.max(Pos[:,2]) / 2 / np.pi
    x,y,z = Pos.transpose()
    z = z/Lz

    Lx = np.max(x)
    lx = np.min(x)
    Ly = np.max(y)
    ly = np.min(y)

    if len(params_helix):
        LX = params_helix[1]
        LY = params_helix[2]
    else:
        LX = Lx-lx
        LY = Ly-ly
    
    x = (x-lx - LX/2) / LX * 2
    y = (y-ly - LY/2) / LY * 2
   
    # #Elliptical Mapping
    x_coord = x * (1-1/2*y**2)**(1/2)
    y_coord = y * (1-1/2*x**2)**(1/2)

    x = x_coord * Lx
    y = y_coord * Ly

    Lims_ellipse = np.array([[np.min(x),np.max(x)],[np.min(y),np.max(y)],[np.min(z),np.max(z)]])
    Pos_elliptical_map = np.array([x,y,z]).transpose()
    # print("RDF ellipse")
    # if metal:
    #         plot_rdf_metal2(Pos_elliptical_map, Types)
    # else:
    #         plot_rdf_sio(Pos_elliptical_map,Types)
    # visualize_cuboid(Pos_elliptical_map, Types)


    ### The helix transformation ###

    #The equations were determined using a 2d roation matrix coupled with a rotation of the basis
    #Another way to see the equations is to compute the tangent and two orthogonal normals to the direction of the 1d helix
    #And to transform the cuboid along these directions : one normal corresponds to y, the other to x and the tangent to x

    R = D * rota
    Norm = (Lz**2 + R**2)**(1/2)
    


    ### STEP 1: reenlarge z ###
    z_coord =  Lz * z + R * x /Norm
    Lims_ellipse = np.array([[np.min(x),np.max(x)],[np.min(y),np.max(y)],[np.min(z_coord),np.max(z_coord)]])
    Pos_elliptical_map = np.array([x,y,z_coord]).transpose()
    print("RDF ellipse change z 2")
    if metal:
            plot_rdf_metal2(Pos_elliptical_map, Types)
    else:
            plot_rdf_sio(Pos_elliptical_map,Types)
    visualize_cuboid(Pos_elliptical_map, Types)


    ### STEP 2: Change y ###
    y_coord = R * np.cos(z*rota) - np.cos(z*rota) * y + Lz * np.sin(z*rota) / Norm * x
    Lims_ellipse = np.array([[np.min(x),np.max(x)],[np.min(y_coord),np.max(y_coord)],[np.min(z_coord),np.max(z_coord)]])
    Pos_elliptical_map = np.array([x,y_coord,z_coord]).transpose()
    # print("RDF ellipse change y 2")
    # if metal:
    #         plot_rdf_metal2(Pos_elliptical_map, Types)
    # else:
    #         plot_rdf_sio(Pos_elliptical_map,Types)
    # visualize_cuboid(Pos_elliptical_map, Types)


    ### STEP 3: change x ###
    x_coord = R * np.sin(z*rota) - np.sin(z*rota) * y - Lz * np.cos(z*rota) / Norm * x
    Lims_ellipse = np.array([[np.min(x_coord),np.max(x_coord)],[np.min(y_coord),np.max(y_coord)],[np.min(z_coord),np.max(z_coord)]])
    # Pos_elliptical_map = np.array([x_coord,y_coord,z_coord]).transpose()
    # print("RDF ellipse change x 2")
    # if metal:
    #         plot_rdf_metal2(Pos_elliptical_map, Types)
    # else:
    #         plot_rdf_sio(Pos_elliptical_map,Types)
    # visualize_cuboid(Pos_elliptical_map, Types)

    #slide automatic
    if slide_z == 0:
        slide_z = np.min(z_coord)
    else:
         z_coord = z_coord - slide_z

    Pos_elliptical_map = np.array([x_coord,y_coord,z_coord]).transpose()
    # print("RDF after slide_z 2")
    # if metal:
    #         plot_rdf_metal2(Pos_elliptical_map, Types)
    # else:
    #         plot_rdf_sio(Pos_elliptical_map,Types, Lims=Lims)

    # visualize_cuboid(Pos_elliptical_map, Types)


    ### STEP 4: add periodic conditions ###
    if do_periodic:
        z_coord = (z_coord < 0) * (z_coord+Lz*2*np.pi) + (z_coord > Lz*2*np.pi) * (z_coord - Lz * 2 * np.pi) + (z_coord >= 0) * (z_coord <= Lz*2*np.pi) * z_coord

    #Slide to the initial mean
    Pos_transfo = np.array([x_coord + mean[0],y_coord + mean[1],z_coord]).transpose()
    Pos_transfo = np.array([x_coord,y_coord,z_coord]).transpose()
    Lims_transfo = np.array([[np.min(Pos_transfo[:,0]-enlarge),np.max(Pos_transfo[:,0])+enlarge],[np.min(Pos_transfo[:,1]-enlarge),np.max(Pos_transfo[:,1])+enlarge],[0,np.max(Pos_transfo[:,2])+enlarge_z]])

    Pos_elliptical_map = np.array([x_coord,y_coord,z_coord]).transpose()
    print("RDF after do_periodic 2")
    if metal:
            plot_rdf_metal2(Pos_elliptical_map, Types, Lims = Lims_transfo)
    else:
            plot_rdf_sio(Pos_elliptical_map,Types, Lims=Lims_transfo)

    visualize_cuboid(Pos_elliptical_map, Types, Lims=Lims_transfo)

    return Pos_transfo, Lims_transfo


















##### ------ Create the helix ------ #####

### RDF check for metal ###


# def remove_boundary_atoms(positions, types, Lims, tol = 1e-6):
#     ax = Lims[0][1] - Lims[0][0]
#     ay = Lims[1][1] - Lims[1][0]
#     az = Lims[2][1] - Lims[2][0]

#     ox = Lims[0][0]
#     oy = Lims[1][0]
#     oz = Lims[2][0]

#     on_boundary = ((np.abs(positions[:,0] - (ox + ax)) < tol)|
#                   (np.abs(positions[:,1] - (oy + ay)) < tol)|
#                   (np.abs(positions[:,2] - (oz + az)) < tol))
#     mask = ~on_boundary
#     return positions[mask], types[mask]


def remove_boundary_atoms(positions, types, Lims_unit, tol=1e-6):
    lx, Lx = Lims_unit[0]
    ly, Ly = Lims_unit[1]
    ax = Lx - lx
    ay = Ly - ly

    # Fractional coordinate within the unit cell
    fx = (positions[:, 0] - lx) / ax
    fy = (positions[:, 1] - ly) / ay

    # Remove atoms sitting on any tiling boundary (fractional coord is
    # a nonzero integer), i.e. duplicates introduced by periodic tiling
    on_boundary = (
        ((np.abs(fx % 1.0) < tol) & (fx > tol)) |
        ((np.abs(fy % 1.0) < tol) & (fy > tol))
    )

    return positions[~on_boundary], types[~on_boundary]

def remove_duplicate_atoms(Pos, Types, tol = 1e-6):
    grid = np.round(Pos/tol).astype(int)
    _, unique_idx = np.unique(grid, axis = 0, return_index=True)
    return Pos[unique_idx], Types[unique_idx]



def plot_rdf_metal2(Pos, Types, type_id=1, cube=100, a_theory=4.078, bins=100, rdf_max = 4.0,
                   periodic=False, Lims=[], vline = 2.95):
    
    d_NN      = a_theory / np.sqrt(2)
    d_2NN = a_theory
    threshold = (d_NN + d_2NN)/2         # upper cutoff: first shell only

    threshold = rdf_max
    Pos_metal = Pos
    N = len(Pos_metal)
    if N == 0:
        print("No atoms with type_id={} found.".format(type_id))
        return

    Lx, Ly, Lz = np.max(Pos_metal, axis=0)
    lx, ly, lz = np.min(Pos_metal, axis=0)

    if periodic:
        if len(Lims) == 0:
            print("No limits provided, running as non-periodic.")
            periodic = False
        else:
            lz, Lz = Lims[2]

    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    Pos_added = np.copy(Pos_metal)
    Idx_added = np.arange(N, dtype = int)


    if periodic:
        Dz       = Lz - lz
        mask_top = Pos_metal[:, 2] > (Lz - threshold)
        mask_bot = Pos_metal[:, 2] < (lz + threshold)
        Pos_added = np.vstack([
            Pos_added,
            Pos_metal[mask_top] - [0, 0, Dz],
            Pos_metal[mask_bot] + [0, 0, Dz],
        ])
        Idx_added = np.concatenate([Idx_added, np.where(mask_top)[0], np.where(mask_bot)[0]])
    

    In_orig = np.zeros(len(Pos_added), dtype=bool)
    In_orig[:N] = True

    Dist_list = []

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):

                Ind_home = (
                    (Pos_added[:, 0] >= x*cube + lx)
                    & (Pos_added[:, 0] <  (x+1)*cube + lx)
                    & (Pos_added[:, 1] >= y*cube + ly)
                    & (Pos_added[:, 1] <  (y+1)*cube + ly)
                    & (Pos_added[:, 2] >= z*cube + lz)
                    & (Pos_added[:, 2] <  (z+1)*cube + lz)
                ) & In_orig

                Ind_pad = (
                    (Pos_added[:, 0] >= x*cube + lx - threshold)
                    & (Pos_added[:, 0] <  (x+1)*cube + lx + threshold)
                    & (Pos_added[:, 1] >= y*cube + ly - threshold)
                    & (Pos_added[:, 1] <  (y+1)*cube + ly + threshold)
                    & (Pos_added[:, 2] >= z*cube + lz - threshold)
                    & (Pos_added[:, 2] <  (z+1)*cube + lz + threshold)
                ) 

                Pos_home = Pos_added[Ind_home]
                Pos_pad = Pos_added[Ind_pad]
                Idx_home = Idx_added[Ind_home]
                Idx_pad = Idx_added[Ind_pad]

                if len(Pos_home) < 2:
                    continue

                D = sd.cdist(Pos_home, Pos_pad)

                for local_i, gi in enumerate(Idx_home):
                    mask = (Idx_pad != gi) & (D[local_i] > 0)
                    # if np.any(mask):
                    #     Dist_list.append([np.min(D[local_i][mask])])
                    Dist_list.append(D[local_i][mask])
                # np.fill_diagonal(D, np.inf)
                # Dist_list.append(D[D < threshold].ravel())

    fig, ax = plt.subplots()
    purple      = np.array([96,  25, 255]) / 255
    dark_purple = np.array([56,  20, 180]) / 255
    dark_dark_purple = np.array([34,  10, 120]) / 255
    Dist_flat = np.concatenate(Dist_list) if Dist_list else np.array([])

    counts, edges, patches =ax.hist(Dist_flat, bins=100, range=(0, rdf_max), color = purple, edgecolor = dark_purple, linewidth = 1)
    radius = ((np.roll(edges, 1) + edges) / 2)[1:]
    dr     = radius[1] - radius[0]


    #Normalisation
    # V = (4/3)*np.pi*(r_max**3)
    # rho = N/V
    # shell_volume = 4*np.pi*radius**2 * dr
    # expected = counts/(2*shell_volume*N)
    # V = (Lims[0][1]-Lims[0][0])*(Lims[1][1]-Lims[1][0])*(Lims[2][1]-Lims[2][0])
    g_r = counts/((4*np.pi*radius**2*dr) * (N)) 

    for rect, val in zip(patches, g_r):
            rect.set_height(val)

    ax.set_title("RDF Au - Au",
                 color=dark_purple)
    ax.set_xlabel("Distance (A)", color=dark_purple)
    ax.set_ylabel("Number of pairs", color=dark_purple)
    ax.set_xticks([k for k in range(int(rdf_max))]+[vline])
    ax.set_xticklabels([k for k in range(int(rdf_max))]+[vline], color = purple)

    ax.tick_params(colors=purple)
    ax.axvline(vline, color=dark_dark_purple)
    ax.set_xlim(0, rdf_max)
    ax.set_ylim(0, 2)
    ax.set_ylabel("g(r)", color = dark_purple)
    plt.tight_layout()
    plt.show()


def plot_rdf_sio(Pos, Types, threshold_Si = 2, threshold_O = 2, rdf_max = 3.2, periodic = False, Lims = [], vline = 1.609, density = True):
    Dist_list, _, _ = compute_hist_neighbors(Pos, Types, threshold_Si=threshold_Si, threshold_O=threshold_O, periodic=periodic, Lims = Lims, rdf_max=rdf_max)
    Dist_list = [k for j in Dist_list for k in j]

    purple      = np.array([96,  25, 255]) / 255
    dark_purple = np.array([56,  20, 180]) / 255
    dark_dark_purple = np.array([34,  10, 120]) / 255

    fig, ax = plt.subplots()
    counts, edges, patches =ax.hist(Dist_list, bins=100, range=(0, rdf_max), color = purple, edgecolor = dark_purple, linewidth = 1)

    if density:
        radius = ((np.roll(edges, 1)+ edges)/2)[1:]
        dr = radius[1] - radius[0]
        # V = (Lims[0][1]-Lims[0][0])*(Lims[1][1]-Lims[1][0])*(Lims[2][1]-Lims[2][0])
        g_r = counts/((4*np.pi*radius**2*dr) * (np.sum(Types == 1))) 
        for rect, val in zip(patches, g_r):
            rect.set_height(val)
        ax.set_ylabel("g(r)", color = dark_purple)
        ax.set_ylim(0, 2)
    else:
        ax.set_ylabel("Number", color = dark_purple)
    
    
    ax.set_title("RDF Si-O",
                 color=dark_purple)
    ax.set_xlabel("Distance (A)", color=dark_purple)
    ax.set_ylabel("Number of pairs", color=dark_purple)
    ax.set_xticks([k for k in range(int(rdf_max))]+[vline])
    ax.set_xticklabels([k for k in range(int(rdf_max))]+[vline], color = purple)

    ax.tick_params(colors=purple)
    ax.axvline(vline, color=dark_dark_purple)
    ax.set_xlim(0, rdf_max)
    plt.tight_layout()
    plt.show()



def visualize_cuboid(Pos, Types, Lims = None, point_size = 8, type_colors = None, short_dist_threshold = 0.0):
    if type_colors is None:
       type_colors = {1: "gold", 2: "red", 3: "blue"}
    plotter = pv.Plotter()
    for t, color in type_colors.items():
        mask = Types == t
        if not np.any(mask):
            continue
        mesh = pv.PolyData(Pos[mask].astype(float))
        plotter.add_mesh(mesh, color=color)

    if Lims is not None:
        box = pv.Box(bounds =(Lims[0][0], Lims[0][1],
                              Lims[1][0], Lims[1][1], 
                              Lims[2][0], Lims[2][1]))
        plotter.add_mesh(box, style = "wireframe")
    plotter.show()
    


def visualize_close_contacts(Pos, Types, Lims=None, threshold=1.0,
                              point_size=8, type_colors=None, au_type=1):
    
    if type_colors is None:
        type_colors = {1: "gold", 2: "red", 3: "blue"}

    plotter = pv.Plotter()

    # ---- render all atoms normally ----
    for t, color in type_colors.items():
        mask = Types == t
        if not np.any(mask):
            continue
        mesh = pv.PolyData(Pos[mask].astype(float))
        plotter.add_mesh(mesh, color=color)

    # ---- find Au-Au close contacts ----
    au_mask  = Types == au_type
    Pos_au   = Pos[au_mask].astype(float)

    if len(Pos_au) > 1:
        tree  = KDTree(Pos_au)
        pairs = np.array(list(tree.query_pairs(r=threshold)), dtype=int)

        if len(pairs):
            # highlight involved atoms (larger, cyan)
            close_idx    = np.unique(pairs.ravel())
            mesh_hi      = pv.PolyData(Pos_au[close_idx])
            plotter.add_mesh(mesh_hi, color="cyan")

            n       = len(pairs)
            pts     = np.vstack([Pos_au[pairs[:, 0]],
                                 Pos_au[pairs[:, 1]]])   # shape (2n, 3)
            cells   = np.empty((n, 3), dtype=int)
            cells[:, 0] = 2
            cells[:, 1] = np.arange(n)
            cells[:, 2] = np.arange(n) + n
            line_mesh        = pv.PolyData()
            line_mesh.points = pts
            line_mesh.lines  = cells.ravel()
            plotter.add_mesh(line_mesh, color="cyan", line_width=1)

            print(f"[close contacts] {n} Au-Au pair(s) with d < {threshold} Å")
            for i, j in pairs:
                d = np.linalg.norm(Pos_au[i] - Pos_au[j])
                print(f"  atom {i} ↔ {j}  :  {d:.4f} Å")
        else:
            print(f"[close contacts] No Au-Au pair closer than {threshold} Å")

    # ---- optional box ----
    if Lims is not None:
        box = pv.Bobounds=(Lims[0][0], Lims[0][1],
                             Lims[1][0], Lims[1][1],
                             Lims[2][0], Lims[2][1])
        plotter.add_mesh(box, style="wireframe")

    plotter.show()

### 

def create_syst(rota,D_exp,pitch,width,thickness,int_thick, asym = 1, 
                do_clean=True,do_periodic=True,circling=True,do_rota_transf=False,file_duplicate="beta_quartz.data",file_output = "quartz_dupl.data", 
                file_output_cast = "quartz_int.data", mass_map = {1: 28.0855, 2: 15.9994, 3: 15.9994, 4: 1.0080}, metal = False, do_angles=False):
    """
    create_syst(rota,D,pitch,width,thickness,int_thick,do_clean=True,do_periodic=True,circling=True,do_rota_transf=False,file_duplicate="beta_quartz.data",do_angles=False)

    This functions creates the whole system using the dimensions stated.
    It creates two files : quartz_dupl.data and quartz_int.data which contains the data of the surface and the data of the cast respectively.

    Parameters
    ----------
        rota : float
            the amount of turns made by the helix
        D_exp : float
            the diameter of the circle containing the helix.
        pitch : float
            the pitch, or the period of the helix
        width : float
            the width of the helix
        thickness : float
            the thickness of the helix
        do_clean : bool, optional
            To clean or not the structure by adding H2O. By default True
        do_periodic : bool, optional
            do the creation of a periodic system after the transformation
        circling : bool, optional
            Circle the basis of the cuboid. This should be used to create helices as is gives much better curvatures. By default True
        do_rota_transf : bool, optional,
            Use the rota transformation instead of the helix one. This should not be used as it gives much worse structure. By default False
        file_duplicate : float, optional,
            the file containing the lattice that will be duplicated
        do_angles : bool, optional
            write the angles SiOH

    Returns
    -------
        Pos_transfo : list
            The position of the atoms of the surface
        Types : list
            The type of the atoms of the surface
        Lims_tot : list
            The lims of the surface
        Angles_OH : list
            The angles SiOH
        Pos_transfo_int : list
            The position of the atoms of the interior
        Types_int : list
            The type of the atoms of the interior
        Lims_tot_int : list
            The lims of the interior

        Two files are also created : quartz_dupl.data and quartz_int.data
    """

    use_cast = True
    Lims, Atom_types, Atom_pos, _ = read_data(file_duplicate,do_scale=False,atom_style="atom")
    print("Read :", file_duplicate)
    
    Lims_visu = [[np.max(Atom_pos[:,0]), np.min(Atom_pos[:,0])], [np.max(Atom_pos[:,1]), np.min(Atom_pos[:,1])], [np.max(Atom_pos[:,2]), np.min(Atom_pos[:,2])]]
    print("First RDF check on the FCC")
    if metal:
            plot_rdf_metal2(Atom_pos, Atom_types, Lims=Lims_visu)
    else:
            plot_rdf_sio(Atom_pos, Atom_types, Lims=Lims_visu)
    visualize_cuboid(Atom_pos, Atom_types, Lims_visu)


    #Get the number of duplication needed to get the proper dimensions
    lx = Lims[0][1] - Lims[0][0]
    ly = Lims[1][1] - Lims[1][0]
    lz = Lims[2][1] - Lims[2][0]

    Nx = int(width // lx +1)
    Ny = int(thickness // ly +1)
    Nz = int(pitch // lz + 1)

    Nx_int = int(int_thick // lx +1)
    Ny_int = int(int_thick // ly +1)

    if asym != 1:
        Nx_left = Nx_int
        Ny_left = Ny_int 

        Nx_right = round(asym*Nx_int)
        Ny_right = round(asym*Ny_int)

        print(Nx-Nx_right)
        print(Ny-Ny_right)

        Nx_list = [0,Nx_left,Nx-Nx_right,Nx]
        Ny_list = [0,Ny_left,Ny-Ny_right,Ny]
    else: 
        print(Nx_int, Ny_int)
        Nx_list = [0,Nx_int,Nx-Nx_int,Nx]
        Ny_list = [0,Ny_int,Ny-Ny_int,Ny]

    N_list = np.array([[Nx],[Ny],[Nz]])

    if int_thick == 0:
        use_cast = False
        Nx_list = Nx
        Ny_list = Ny

    print(Nx_list,Ny_list,Nz)

    ##Due to the transformation, the D is not the same as the one experimentally
    #It is however possible to find the D corresponding to the transformation

    #The maximum distance is taken on the ellipse
    W = width
    P = pitch/2/np.pi
    T = thickness


    D_est = (-T+D_exp) /2
    N = P / (P*P + D_est*D_est)**(1/2)
    b = abs(2*T*D_est / (W*W*N*N - T*T))
    # print(b)

    if b > 1 :
        #The extremum point is the point in the external layer
        D_transfo = D_est
    else:
        #The extremum is in-between
        d0 = (W*W/4 - D_exp**2/4) * P**4 * (W*W-T*T)
        d2 = (P*P*D_exp**2/4 * (T*T-W*W) + P*P*T*T * (D_exp**2/4-W*W/4)) + P**4*W*W
        d4 = (D_exp**2/4 *T*T + W*W*P*P)
        delta = (d2**2 - 4*d4*d0)
        if delta > d2**2:
            D_transfo = ((-d2 + delta**(1/2))/2/d4)**(1/2)  
        elif delta < 0:
            print("Two possiblities for the D, the higher one has been taken")
            D_transfo = np.real(((-d2 + delta**(1/2))/2/d4)**(1/2))
        else:
            print("Two possiblities for the D, the higher one has been taken")
            D_transfo = ((-d2 - delta**(1/2))/2/d4)**(1/2)
            D_transfo = ((-d2 + delta**(1/2))/2/d4)**(1/2)

    print("Old D", D_transfo,D_exp)

    ##Surface
    if do_clean:
        ## For oxydes --> need to "clean" the structure with OH bonds
        #Create one slab that is corrected then duplicated Nz times
        #The slab contains 3 duplicates as to only take the inside when doing the cleaning
        Pos, Types, Lims_tot, _a, _b = duplicate(Nx,Ny,3,Lims,Atom_types,Atom_pos)
        Pos, Types, Bonds_OH, Angles_OH = clean_structure(Pos,Types,Lims,N_list,periodic=True)
        Pos, Types, Lims_tot, Bonds_OH, Angles_OH = duplicate(1,1,Nz,Lims,Types,Pos,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH)

        print("2nd RDF check before the transformation")
        plot_rdf_sio(Pos, Types)
        visualize_cuboid(Pos, Types)
        if not do_angles:
            Angles_OH = []
    else:
        ## Create slab form which we remove the duplicate atoms
        Pos, Types, Lims_tot, _a, _b = duplicate(Nx, Ny, Nz,Lims,Atom_types,Atom_pos)
        # Pos, Types = remove_boundary_atoms(Pos, Types, Lims_tot)
        # Pos, Types = remove_duplicate_atoms(Pos, Types)
        
        # Pos_slab, Types_slab, Lims_tot, _a, _b = duplicate(Nx,Ny,3,Lims,Atom_types,Atom_pos)
        # Middle = ((Pos_slab[:,2] >= (Lims[2][1]+Lims[2][0]))&
        #           (Pos_slab[:,2] < (Lims[2][1]*2+Lims[2][0])))
        # Pos_clean = Pos_slab[Middle]
        # Types_clean = Types_slab[Middle]
        # Pos_clean = Pos_clean - np.array([0,0, np.min(Pos_clean[:,2])])

        # Pos, Types, Lims_tot, Bonds_OH, Angles_OH = duplicate(1,1,Nz,Lims,Types_clean,Pos_clean)

        Lims_visu = [[np.max(Pos[:,0]), np.min(Pos[:,0])], [np.max(Pos[:,1]), np.min(Pos[:,1])], [np.max(Pos[:,2]), np.min(Pos[:,2])]]
        print("2nd RDF check before the transformation")
        plot_rdf_metal2(Pos, Types)
        visualize_cuboid(Pos, Types)

        Bonds_OH, Angles_OH = [], []



    ## Add cast to relax against during the dynamic
    if use_cast:
        print("Separation to make cast")
        # Mask to separate the surface and the cast later
        ## Need elliptical mapping for this ##
        mask_int_x = (((Pos[:,0] - Lims[0][0])//lx >= Nx_list[1]) & ((Pos[:,0] - Lims[0][0])//lx < Nx_list[2]))
        mask_int_y = (((Pos[:,1] - Lims[1][0])//ly >= Ny_list[1]) & ((Pos[:,1] - Lims[1][0])//ly < Ny_list[2]))

        mask_int = mask_int_x & mask_int_y
        mask_surf = ~mask_int
        
        # Transformation of the cuboid into the helice
        Pos_transfo,Lims_tot = transfo(Pos, Types, Lims, mass_map, metal = metal, D=D_transfo,rota=rota,do_periodic=do_periodic,params_helix=[pitch,width,thickness])
        if metal:
            plot_rdf_metal2(Pos_transfo, Types)
        else:
            plot_rdf_sio(Pos_transfo,Types)
        visualize_cuboid(Pos_transfo, Types)

        # Mapping of the indices between cast and surface
        Pos_transfo_int = Pos_transfo[mask_int]
        Types_int = Types[mask_int]
        print("RDF pos_transfo int")
        if metal:
            plot_rdf_metal2(Pos_transfo_int, Types_int)
        else:
            plot_rdf_sio(Pos_transfo_int,Types_int)
        visualize_cuboid(Pos_transfo_int, Types_int)
        

        Pos_transfo_surf = Pos_transfo[mask_surf]
        Types_surf = Types[mask_surf]
        print("RDF pos_transfo surf")
        if metal:
            plot_rdf_metal2(Pos_transfo_surf, Types_surf)
        else:
            plot_rdf_sio(Pos_transfo_surf,Types_surf)
        visualize_cuboid(Pos_transfo_surf, Types_surf)
        
        Ind_cut = mask_surf
        Ind_sum = np.cumsum(Ind_cut) - 1
        new_bond_surf = []

        # Recovery of OH Bond indices 
        for a1,a2 in Bonds_OH:
            i1 = a1 - 1
            i2 = a2 - 1   
            if mask_surf[i1] and mask_surf[i2]:
                new_a1 = Ind_sum[i1] + 1
                new_a2 = Ind_sum[i2] + 1
                new_bond_surf.append([new_a1, new_a2])

        for a1, a2 in new_bond_surf:
            d = np.linalg.norm(Pos_transfo_surf[a2 - 1] - Pos_transfo_surf[a1 - 1])
            Lz = Lims_tot[2][1] - Lims_tot[2][0]

            delta = Pos_transfo_surf[a2 - 1,2] - Pos_transfo_surf[a1 - 1, 2]
            delta -= Lz*np.round(delta/Lz)
            dmin = np.linalg.norm(delta)
            if d > 10: # Check if bond too large
                dz = Pos_transfo_surf[a2 - 1, 2] - Pos_transfo_surf[a1 - 1, 2]
                print(dz, Lz/2)
                if dz > Lz/2: # Check if pbc respected 
                    Pos_transfo_surf[a2 - 1,2] -= Lz
                if dz < -Lz/2:
                    Pos_transfo_surf[a2 - 1,2] += Lz
                print(Pos_transfo_surf[a2 - 1, 2], Lz/2)
                d = np.linalg.norm(Pos_transfo_surf[a1 - 1] - Pos_transfo_surf[a2 - 1])

        # Write data files
        if Pos_transfo_surf.size > 0:
            write_data(file_output,Pos_transfo_surf,Types_surf,Lims_tot,Bonds_OH=new_bond_surf,Angles_OH=Angles_OH, mass_map=mass_map)
            print("Written: ", file_output)
        else:
            print("surface is empty")

        write_data(file_output_cast,Pos_transfo_int,Types_int,Lims_tot,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH, mass_map=mass_map)
        print("Written: ", file_output_cast)
    
       
    ##If we don't add a cast to relax against
    else:
        print("no cast")
        ##Transformation of the cuboid into the helice

        Pos_transfo,Lims_tot = transfo(Pos,Types, Lims_tot, mass_map, metal = metal, D=D_transfo,rota=rota,do_periodic=do_periodic,params_helix=[pitch,width,thickness])   
        Pos_transfo_surf,Types_surf = remove_duplicate_atoms(Pos_transfo, Types)
        Pos_transfo_int = []
        Types_int = []

        visualize_close_contacts(Pos_transfo_surf, Types_surf)

        Lims_visu = [[np.max(Pos_transfo_surf[:,0]), np.min(Pos_transfo_surf[:,0])], [np.max(Pos_transfo_surf[:,1]), np.min(Pos_transfo_surf[:,1])], [np.max(Pos_transfo_surf[:,2]), np.min(Pos_transfo_surf[:,2])]]
        print("3rd RDF check after the removal of duplicates")
        if metal:
            plot_rdf_metal2(Pos_transfo_surf, Types_surf)
        else:
            plot_rdf_sio(Pos_transfo_surf,Types_surf)
        visualize_cuboid(Pos_transfo_surf, Types_surf)


        # D = sd.cdist(Pos_transfo_surf, Pos_transfo_surf)
        # np.fill_diagonal(D, np.inf)
        # close = np.argwhere((D<= 0.5))
        # for i,j in close:
        #     if i < j:
        #         Lz = Lims_tot[2][1] - Lims_tot[2][0]
        #         d = np.linalg.norm(Pos_transfo_surf[i] - Pos_transfo_surf[j])
        #         if d < 2.0: # Check if bond too large
        #             print("distance", D[i,j])
        #             print("Pos i, Pos j", Pos[i], Pos[j])
        #             print("Pos transfo i, Pos transfo j", Pos_transfo_surf[i], Pos_transfo_surf[j])
        #             print("dist bef", np.linalg.norm(Pos[i] - Pos[j]))
        #             print("delta z bef", abs((Pos[i, 2] - Pos[j, 2])))
        #             print("delta z aft", abs((Pos_transfo_surf[i, 2] - Pos_transfo_surf[j, 2])))
                    

        write_data(file_output,Pos_transfo_surf,Types_surf,Lims_tot,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH, mass_map=mass_map)
        

    return Pos_transfo_surf, Types_surf, Lims_tot, Angles_OH, Pos_transfo_int, Types_int, Lims_tot



 # Pos, Types, Lims_tot, _a, _b = duplicate(Nx_list,Ny_list,Nz,Lims,Atom_types,Atom_pos)
        # Bonds_OH, Angles_OH = [], []
        # Pos_transfo,Lims_tot,slide_z = transfo(Pos,Types, Lims, mass_map, D=D_transfo,rota=rota,do_periodic=do_periodic,params_helix=[pitch,width,thickness])
        # write_data("quartz_dupl.data",Pos_transfo,Types,Lims_tot,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH)

        # ##Inside
        # if not type(Nx_list) is int:
        #     Nx_int = Nx_list[1:3]
        #     Ny_int = Ny_list[1:3]
        #     Pos_int, Types_int, Lims_tot_int, _a, _b = duplicate(Nx_int,Ny_int,Nz,Lims,Atom_types,Atom_pos)
        #     Pos_transfo_int ,Lims_tot_int,slide_z = transfo(Pos_int,Types_int, Lims_tot_int,mass_map, D=D_transfo,rota=rota,slide_z=slide_z,do_periodic=do_periodic)
        #     write_data("quartz_int.data",Pos_transfo_int,Types_int,Lims_tot_int,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH)
        # else:
        #     Pos_transfo_int = None
        #     Types_int = None
        #     Lims_tot_int = None
        # return Pos_transfo, Types, Lims_tot, Angles_OH, Pos_transfo_int, Types_int, Lims_tot_int

            