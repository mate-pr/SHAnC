"""
Helix System Creation for SHAnC

distortion.py contains tools to duplicate lattice slabs, clean oxide surfaces,
transform a slab into a helical geometry, and generate full periodic systems
for simulation. The functions are specialized for oxide/metal structures and
maintain bond and angle connectivity for OH-passivated surfaces.

"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Construction.import_libraries import *
from Construction.read_write import *
from Construction.analysis import *

# distortion.py contains tools to duplicate lattice slabs, clean oxide surfaces,
# transform a slab into a helical geometry, and generate full periodic systems
# for simulation. The functions are specialized for oxide/metal structures and
# maintain bond and angle connectivity for OH-passivated surfaces.

# ---------------------------------------------------------------------------
# Duplication of the lattice
# ---------------------------------------------------------------------------

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

    # Support either an integer replicate count or a list describing a
    # hollow/partial replication pattern. This makes it possible to generate
    # full slabs, hollow beams, or custom surface geometries.
    flag_x = 0
    flag_y = 0
    if type(Nx_list) is int:
        Nx = Nx_list
        flag_x = 1

    elif len(Nx_list) == 2:
        Nx_0 = Nx_list[0]
        Nx = Nx_list[1]
        flag_x = 1

    else:
        Nx = Nx_list[-1]

    if type(Ny_list) is int:
        Ny = Ny_list
        flag_y = 1

    elif len(Ny_list) == 2:
        Ny_0 = Ny_list[0]
        Ny = Ny_list[1]
        flag_y = 1

    else:
        Ny = Ny_list[-1]

    for x in range(Nx_0, Nx):
        for y in range(Ny_0,Ny):
            # Determine whether the current replicate cell should be included.
            # Full-int replication occurs when a scalar Nx or Ny is used; for
            # list-based hollow structures, include only the specified ranges.
            if flag_x or flag_y or (x >= Nx_list[0] and x < Nx_list[1]) or (x >= Nx_list[2] and x <= Nx_list[3]) or (y >= Ny_list[0] and y < Ny_list[1]) or (y >= Ny_list[2] and y <= Ny_list[3]):
                for z in range(Nz):
                    for at, ty, num in zip(Atom_pos, Atom_type, range(len(Atom_pos))):
                        Pos.append(at + np.array([x*Lx+lx, y*Ly+ly, z*Lz+lz]))
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


# ---------------------------------------------------------------------------
# Structure Passivation / Cleaning
# ---------------------------------------------------------------------------

def _find_missing_bond_dir(bonded_vecs, sign_x_fallback):
    """
    Return the normalised direction of the most likely missing bond,
    given the vectors from a cation to its currently bonded anions.

    Method: the missing bond points opposite to the centroid of the
    existing bond unit-vectors.  This is exact for symmetric polyhedra
    (tetrahedral Si, octahedral Ti) that have lost exactly one bond.

    Falls back to the Â±x axis when the existing bonds cancel out exactly
    (fully symmetric residual â€” cannot determine a unique direction).

    Parameters
    ----------
    bonded_vecs      : np.ndarray (K, 3)  vectors Cat â†’ bonded O
    sign_x_fallback  : Â±1  used when the heuristic fails

    Returns
    -------
    direction : np.ndarray (3,)  normalised
    """
    if len(bonded_vecs) == 0:
        return np.array([float(sign_x_fallback), 0.0, 0.0])

    norms = np.linalg.norm(bonded_vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)   # avoid div-by-zero
    unit_vecs = bonded_vecs / norms

    missing = -np.sum(unit_vecs, axis=0)
    mag     = np.linalg.norm(missing)

    if mag < 0.1:
        # Bonds cancel --> cannot determine unique missing direction
        return np.array([float(sign_x_fallback), 0.0, 0.0])

    return missing / mag


def clean_structure(Pos, Types, Lims, N, periodic=True,
                        cation_type=1, anion_type=2,
                        expected_coord=4,
                        bond_threshold=2.0,
                        bond_length=1.6, h_length=1.0,
                        collision_tol=0.8, min_count = None): 
    """
    Cleans the surface of an oxide slab by adding OH groups to under-
    coordinated cations, generalised to any cation type.
    """
    oh_length = bond_length + h_length 
    Center = [
    (Lims[0][1]*N[0][0] - Lims[0][0]) / 2,
    (Lims[1][1]*N[1][0] - Lims[1][0]) / 2,
    (Lims[2][1]*N[2][0] - Lims[2][0]) / 2,
    ]

    Pos_Cat = Pos[Types == cation_type]
    Pos_An = Pos[Types == anion_type]
    
    D_all = sd.cdist(Pos_Cat, Pos_An) # (n_cat, n_an)
    Cat_count_An = np.sum(D_all < bond_threshold, axis=1)

    Lz_unit = Lims[2][1] - Lims[2][0]
    z_mid_lo = Lims[2][0] + Lz_unit
    z_mid_hi = Lims[2][0] + 2*Lz_unit

    mid_mask = (Pos_Cat[:,2] >= z_mid_lo) & (Pos_Cat[:,2]< z_mid_hi)
    if min_count is None:
        min_count = int(np.min(Cat_count_An[mid_mask]))
    else:
        min_count = min_count

    if np.min(Cat_count_An) >= expected_coord:
        print("All cations fully coordinated.")
        return Pos, Types, [], []


    Lack_Cat_mask = (Cat_count_An <= 2) 
    Lack_Cat = Pos_Cat[Lack_Cat_mask]
    Lack_Cat_D_rows = D_all[Lack_Cat_mask] 

    print(f"bond_threshold={bond_threshold} angstrom, "
    f"expected_coord={expected_coord}, collision_tol={collision_tol} angstrom")
    print(f" bond count distribution: "
    f"{dict(zip(*np.unique(Cat_count_An, return_counts=True)))}")
    print(f" minimum bond count = {min_count} "
    f"{len(Lack_Cat)} surface cation(s) to cap")

    Atoms_add_pos = []
    Atoms_add_types = []
    Bonds_OH = []
    num_at = len(Pos)
    for Cat, D_row in zip(Lack_Cat, Lack_Cat_D_rows):

        sign_x = np.sign(Cat[0] - Center[0])
        if sign_x == 0:
            sign_x = 1

        # Bonded-O vectors : used by _find_missing_bond_dir if collision fires
        bonded_vecs = Pos_An[D_row < bond_threshold] - Cat # (K, 3)

        # Add OH on the outward face of Cat 
        Atom_O_candidate = Cat + sign_x * np.array([bond_length, 0.0, 0.0])

        # We rebuild Pos_An_live each iteration because Types may have changed
        # (earlier iterations convert type-2 O to type-3 Oh via Types[idx]=3).
        Lz_slab = (Lims[2][1] - Lims[2][0])
        Pos_An_live = Pos[(Types == anion_type) | (Types == 3)]
        Pos_An_check = np.vstack([Pos_An_live, Pos_An_live + [0, 0, Lz_slab],
                                  Pos_An_live + [0, 0, -Lz_slab]])
        if len(Pos_An_live) > 0:
            # Check that the candidate OH position does not collide with existing O/H
            # atoms, including periodic copies in z.
            dist_cand = np.linalg.norm(Pos_An_check - Atom_O_candidate, axis=1)
            collision = float(np.min(dist_cand)) < collision_tol
        else:
            collision = False

        if collision:
            place_dir = _find_missing_bond_dir(bonded_vecs, sign_x)
        else:
            place_dir = np.array([float(sign_x), 0.0, 0.0])

        place_dir[2] = 0.0
        mag = np.linalg.norm(place_dir)
        if mag < 1e-10:
            place_dir = np.array([float(sign_x), 0.0, 0.0])
        else:
            place_dir = place_dir/mag

        Atom_O_candidate2 = Cat + place_dir*bond_length
        dist_cand2 = np.linalg.norm(Pos_An_check-Atom_O_candidate2, axis = 1)

        if float(np.min(dist_cand2)) < collision_tol:
            print("increase bond length or collision_tol")

        
        Atom_O_add = Cat + place_dir * bond_length
        Atom_H_add = Cat + place_dir * oh_length
        Atoms_add_pos.extend([Atom_O_add, Atom_H_add])
        Atoms_add_types.extend([3, 4])
        Bonds_OH.append([num_at, num_at + 1])

        # Add an H on the opposite surface  
        Cat_symm = np.array([2 * Center[0] - Cat[0], Cat[1], Cat[2]])

        Pos_An_trunc = Pos_An[
            (Pos_An[:, 2] > (Cat[2] - 2)) & (Pos_An[:, 2] < (Cat[2] + 2))
        ]
        D_to_symm = np.sum((Pos_An_trunc - Cat_symm.reshape((1, 3))) ** 2, axis=1)
        # Simply take the O closest to the symmetric point of Cat 
        Pos_An_cap = Pos_An_trunc[np.argmin(D_to_symm)]

        sign_cap = np.sign(Pos_An_cap[0] - Center[0])
        if sign_cap == 0:
            sign_cap = -sign_x
        Atom_H_cap = Pos_An_cap + sign_cap * np.array([h_length, 0.0, 0.0])
        Atoms_add_pos.append(Atom_H_cap)
        Atoms_add_types.append(4)

        D_global_all = np.sum((Pos - Pos_An_cap) ** 2, axis=1)
        idx_cap = np.argmin(D_global_all)
        Types[idx_cap] = 3
        Bonds_OH.append([idx_cap, num_at + 2])
        num_at += 3

    if len(Atoms_add_pos) == 0:
        print("No under-coordinated cations found.")
        return Pos, Types, [], []

    Atoms_add_pos = np.array(Atoms_add_pos, dtype=float)
    Atoms_add_types = np.array(Atoms_add_types, dtype=int)

    New_Pos = np.append(Pos, Atoms_add_pos, axis=0)
    New_Types = np.append(Types, Atoms_add_types, axis=0)

    Middle = (
    (New_Pos[:, 2] >= (Lims[2, 1] + Lims[2, 0]))
    & (New_Pos[:, 2] < (Lims[2, 1]*2 + Lims[2, 0]))
    )
    New_Pos = New_Pos[Middle]
    New_Pos = New_Pos - np.array([0, 0, np.min(New_Pos[:, 2])])
    New_Types = New_Types[Middle]

    Bonds_OH_corrected = []
    New_index = (np.cumsum(Middle) - 1) * Middle

    for bond in Bonds_OH:
        if New_index[bond[0]] != 0 and New_index[bond[1]] != 0:
            Bonds_OH_corrected.append(
            [New_index[bond[0]] + 1, New_index[bond[1]] + 1]
            )

    Angles_OH = []
    Pos_Cat_new = New_Pos[New_Types == cation_type]

    for bond in Bonds_OH_corrected:
        O = New_Pos[bond[0] - 1]
        Pos_Cat_trunc = Pos_Cat_new[
        (Pos_Cat_new[:, 2] > (O[2] - 2)) & (Pos_Cat_new[:, 2] < (O[2] + 2))
        ]
    
        Dist_Cat = np.sum((Pos_Cat_trunc - O.reshape((1, 3)))**2, axis=1)
        Index_Cat_trunc = (
        (New_Types == cation_type)
        & (New_Pos[:, 2] > (O[2] - 2))
        & (New_Pos[:, 2] < (O[2] + 2))
        )
        Index_Cat_bond = np.argmax((np.cumsum(Index_Cat_trunc) - 1) == np.argmin(Dist_Cat)
        )
        Angles_OH.append([Index_Cat_bond + 1, bond[0], bond[1]])

    return New_Pos, New_Types, Bonds_OH_corrected, Angles_OH

# ---------------------------------------------------------------------------
# Geometric Helical transformation
# ---------------------------------------------------------------------------

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

    x,y,z = Pos.transpose()
    Lx = np.max(x)
    lx = np.min(x)
    Ly = np.max(y)
    ly = np.min(y)

    if len(params_helix):
        LX = params_helix[1]
        LY = params_helix[2]
        Lz = params_helix[0] / 2 / np.pi
    else:
        LX = Lx-lx
        LY = Ly-ly
        Lz = np.max(Pos[:,2]) / 2 / np.pi

    z = z/Lz
    x = (x-lx - LX/2) / LX * 2
    y = (y-ly - LY/2) / LY * 2
   

    ## Elliptical Mapping
    x_coord = x * (1-1/2*y**2)**(1/2)
    y_coord = y * (1-1/2*x**2)**(1/2)


    ## Rescaling 
    x = x_coord * Lx
    y = y_coord * Ly

    Pos_elliptical_map = np.array([x,y,z]).transpose()

    ### The helix transformation ###

    # The equations were determined using a 2d roation matrix coupled with a rotation of the basis
    # Another way to see the equations is to compute the tangent and two orthogonal normals to the direction of the 1d helix
    # And to transform the cuboid along these directions : one normal corresponds to y, the other to x and the tangent to x

    R = D * rota
    Norm = (Lz**2 + R**2)**(1/2)

    ## STEP 1: Resizing of z and tilting of the base ##
    z_coord = Lz*z + R * x /Norm
    Pos_elliptical_map = np.array([x,y,z_coord]).transpose()
  
    ## STEP 2: Transformation along y ##
    y_coord = R * np.cos(z*rota) - np.cos(z*rota) * y + Lz * np.sin(z*rota) / Norm * x

    ## STEP 3: Transformation along x ##
    x_coord = R * np.sin(z*rota) - np.sin(z*rota) * y - Lz * np.cos(z*rota) / Norm * x
    
    if slide_z == 0:
        slide_z = np.min(z_coord)
    else:
         z_coord = z_coord - slide_z

    ## STEP 4: Add periodic conditions ##
    if do_periodic: 
        z_coord = (z_coord < 0) * (z_coord+Lz*2*np.pi) + (z_coord > Lz*2*np.pi) * (z_coord - Lz * 2 * np.pi) + (z_coord >= 0) * (z_coord <= Lz*2*np.pi) * z_coord

    # Slide to the initial mean
    # Build the transformed coordinates and output bounding box with the
    # requested padding around the helix.
    Pos_transfo = np.array([x_coord, y_coord, z_coord]).transpose()
    Lims_transfo = np.array([
        [np.min(Pos_transfo[:, 0] - enlarge), np.max(Pos_transfo[:, 0]) + enlarge],
        [np.min(Pos_transfo[:, 1] - enlarge), np.max(Pos_transfo[:, 1]) + enlarge],
        [0, np.max(Pos_transfo[:, 2]) + enlarge_z],
    ])

    return Pos_transfo, Lims_transfo



# ---------------------------------------------------------------------------
# System generation 
# ---------------------------------------------------------------------------


def create_syst(rota,D_exp,pitch,width,thickness,int_thick, asym = 1, Twist = False, 
                do_clean=True,do_periodic=True, file_duplicate="beta_quartz.data",file_output = "quartz_dupl.data", 
                file_output_cast = "quartz_int.data", mass_map = {1: 28.0855, 2: 15.9994, 3: 15.9994, 4: 1.0080}, metal = False):
    
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
    
    # Get the number of replication steps needed to fill the requested geometry.
    # Nx and Ny set the slab width/thickness, Nz sets the pitch-length along z.
    lx = Lims[0][1] - Lims[0][0]
    ly = Lims[1][1] - Lims[1][0]
    lz = Lims[2][1] - Lims[2][0]

    Nx = int(width // lx + 1)
    Ny = int(thickness // ly + 1)
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

    if not metal:
        Atom_pos[:,2] += lz/4
        Atom_pos[:, 2] %= lz

    supercell_Lx = Nx*lx
    supercell_Ly = Ny*ly
    supercell_Lz = Nz*lz

    print("Supercell dimensions :", supercell_Lx, supercell_Ly, supercell_Lz)
    print("The unit cell is repeated",Nx, Ny, Nz,"times") 
    
    ## Due to the transformation, the D is not the same as the one experimentally
    # It is however possible to find the D corresponding to the transformation

    # The maximum distance is taken on the ellipse
    W = width
    P = pitch/2/np.pi
    T = thickness


    D_est = (-T+D_exp) /2
    N = P / (P*P + D_est*D_est)**(1/2)
    b = abs(2*T*D_est / (W*W*N*N - T*T))
    # print(b)

    if b > 1 :
        # The extremum point is the point in the external layer
        D_transfo = D_est
    else:
        # The extremum is in-between
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

    if D_transfo < 0:
        D_transfo = 0

    if Twist:
        D_transfo = 0

    ## Surface generation
    if do_clean:
        ## For oxides: clean the slab by adding OH passivation on undercoordinated cations.
        # Build a temporary slab with an extra duplicate region so the surface cleaning
        # can be applied while preserving only the interior after capping.
        Pos, Types, Lims_tot, _a, _b = duplicate(Nx, Ny, 3, Lims, Atom_types, Atom_pos)
        Pos, Types, Bonds_OH, Angles_OH = clean_structure(Pos, Types, Lims, N_list, periodic=True)
        Pos, Types, Lims_tot, Bonds_OH, Angles_OH = duplicate(1, 1, Nz, Lims, Types, Pos, Bonds_OH=Bonds_OH, Angles_OH=Angles_OH)

        # Adapt the pitch to the new z dimension to avoid problems with the periodicity of the system after the transformation
        pitch_actual = Nz*lz
        Pos[:,2] *=pitch/pitch_actual
        Lims_tot[2] *=pitch/pitch_actual
        write_data("cuboid.data", Pos, Types, Lims=Lims_tot, Bonds_OH=Bonds_OH, mass_map=mass_map)
        
    else:
        ## Create slab form which we remove the duplicate atoms ##
        Pos, Types, Lims_tot, _a, _b = duplicate(Nx, Ny, Nz,Lims,Atom_types,Atom_pos)
        
        if not metal:
            print(Lims_tot)
        
            pitch_actual = Nz*lz
            Pos[:,2] *=pitch/pitch_actual
            Lims_tot[2] *=pitch/pitch_actual

    
        Bonds_OH, Angles_OH = [], []
        Lims_visu = [[0,Nx*lx], [0, Ny*ly], [0, Nz*lz]]
        write_data("cuboid.data", Pos, Types, Lims=Lims_visu, Bonds_OH=Bonds_OH, mass_map=mass_map)
        



    ## Add cast to relax against during the dynamic
    if use_cast:
        print("Separation between helix wall and cast")
        # Mask to separate the surface and the cast later
        ## Need elliptical mapping for this ##
        mask_int_x = (((Pos[:,0] - Lims[0][0])//lx >= Nx_list[1]) & ((Pos[:,0] - Lims[0][0])//lx < Nx_list[2]))
        mask_int_y = (((Pos[:,1] - Lims[1][0])//ly >= Ny_list[1]) & ((Pos[:,1] - Lims[1][0])//ly < Ny_list[2]))

        mask_int = mask_int_x & mask_int_y
        mask_surf = ~mask_int
        
        # Transformation of the cuboid into the helice
        Pos_transfo,Lims_tot = transfo(Pos, Types, Lims, mass_map, metal = metal, D=D_transfo,rota=rota,do_periodic=do_periodic,params_helix=[pitch,width,thickness])
        import pyvista as pv 
        pv.global_theme.jupyter_backend = "None"
        visualize_structure(Pos_transfo, Types)

        # Map original slab atoms to the transformed cast and surface partitions.
        Pos_transfo_int = Pos_transfo[mask_int]
        Types_int = Types[mask_int]
        print("Cast after transformation")
        visualize_structure_cast_surface_separation(Pos_transfo_int, Types_int, cast=True)
        
        Pos_transfo_surf = Pos_transfo[mask_surf]
        Types_surf = Types[mask_surf]
        print("Surface after transformation")
        visualize_structure_cast_surface_separation(Pos_transfo_surf, Types_surf, cast=False)
        
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

        # Check if the bonds are not too large after the transformation and if the pbc are respected, 
        # if not adapt the position of the atoms
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
        else:
            print("Surface is empty")

        write_data(file_output_cast,Pos_transfo_int,Types_int,Lims_tot,Bonds_OH=Bonds_OH,Angles_OH=Angles_OH, mass_map=mass_map)
    
    ## If we don't add a cast to relax against ##
    else:
        print("No cast added, the helix is full")
        ## Transformation of the cuboid into the helice ##
        Pos_transfo,Lims_tot = transfo(Pos,Types, Lims_tot, mass_map, metal = metal, D=D_transfo,rota=rota,do_periodic=do_periodic,params_helix=[pitch,width,thickness])   
        Pos_transfo_int = []
        Types_int = []
        Pos_transfo_surf, Types_surf = Pos_transfo, Types

        visualize_structure(Pos_transfo_surf, Types_surf)
        write_data(file_output, Pos_transfo_surf, Types_surf, Lims_tot,
                Bonds_OH=Bonds_OH, Angles_OH=Angles_OH, mass_map=mass_map)

    return Pos_transfo_surf, Types_surf, Lims_tot, Angles_OH, Pos_transfo_int, Types_int, Lims_tot