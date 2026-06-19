"""
Chromophore attachment and surface extraction utilities for SHAnC.

This module provides tools for:
  - locating surface and anchor sites on silica-based systems
  - estimating surface area using slice contours and perimeter methods
  - grafting molecule anchors and chromophores to surfaces
  - preparing SDF and XYZ molecules for attachment

It uses ``read_write.py`` for file I/O, ``distortion.py`` for geometric
transformations, and ``analysis.py`` for neighbor-count based surface
validation.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Construction.import_libraries import *
from Construction.distortion import *
from Construction.read_write import *
from Construction.analysis import *

BOND_TYPE_MAP = {
    (frozenset({'C', 'H'}), 1): 1,
    (frozenset({'C', 'C'}), 1): 1,
    (frozenset({'C', 'C'}), 2): 2,
    (frozenset({'C', 'C'}), 3): 3,
    (frozenset({'C', 'O'}), 1): 1,
    (frozenset({'C', 'O'}), 2): 2,
    (frozenset({'C', 'N'}), 1): 1,
    (frozenset({'N', 'H'}), 1): 1,
}




SI_C_BOND_TYPE = 100
AMIDE_BOND_TYPE = 101




# ---------------------------------------------------------------------------
# Surface calculation helpers
# ---------------------------------------------------------------------------

def _slice_z(Pos, z0, dz):
    """Return a boolean mask selecting atoms within a z-slice."""
    mask = (Pos[:, 2] >= z0 - dz / 2) & (Pos[:, 2] <= z0 + dz / 2)
    return mask

def circle(Pos, z0, dz, center_mode="mid", shift=False):
    """Return the slice mask, center, and radius for a z-plane cross section."""

    mask = _slice_z(Pos, z0, dz)
    Pos_slice = Pos[mask]
    if len(Pos_slice) == 0:
        return None, None, None

    xy = Pos_slice[:, :2]

    if center_mode == "mean":
        center = np.mean(xy, axis=0)
    elif center_mode == "mid":
        center = (np.max(xy, axis=0) + np.min(xy, axis=0)) / 2
    else:
        raise ValueError("center_mode must be 'mid' or 'mean'")

    distances = np.linalg.norm(xy - center, axis=1)
    R = np.max(distances)

    if shift:
        center = np.mean(Pos[:, :2], axis=0)
    return mask, center, R

def circle_min_distances(Pos_slice, center, R, n_points=500):
    """Sample circle positions and return nearest atom distances."""
    xy = Pos_slice[:, :2]

    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle_x = center[0] + R * np.cos(theta)
    circle_y = center[1] + R * np.sin(theta)
    circle_pts = np.column_stack((circle_x, circle_y))

    diff = circle_pts[:, None, :] - xy[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    min_dist = np.min(dist, axis=1)
    closest_indices = np.argmin(dist, axis=1)

    return circle_pts, closest_indices, min_dist, theta

def second_circle(circle_pts, min_dist, theta):
    """Return the center and radius of the second contour circle."""
    idx_max = np.argmax(min_dist)
    center2 = circle_pts[idx_max]
    r2 = min_dist[idx_max]
    return center2, r2

def external_surface(Pos, list_BOX, dz=1.0, n_points=1000, twist=False):
    """Identify external surface contour atoms using slice-based circles."""
    z_min = list_BOX[0][2][0]
    z_max = list_BOX[0][2][1]
    contour_indices_global = []

    z_values = np.arange(0, z_max, dz)
    contour_per_slice = []
    for z0 in z_values:
        mask = _slice_z(Pos, z0, dz)
        Pos_slice = Pos[mask]
        
        if len(Pos_slice)==0:
            continue

        mask_slice, center1, R1 = circle(Pos_slice, z0, dz)
        if mask_slice is None:
            continue
        
        circle_pts1, closest_indices1, min_dist1, theta1 = circle_min_distances(Pos_slice, center1, R1, n_points=n_points)
        if not twist:
            center2, R2 = second_circle(circle_pts1, min_dist1, theta1)
            circle_pts2, closest_indices2, min_dist2, theta2 = circle_min_distances(Pos_slice, center2, R2, n_points=n_points)
            slice_indices = np.unique(np.concatenate([closest_indices1, closest_indices2]))
        else:
              slice_indices = np.unique(np.concatenate([closest_indices1]))

        global_indices = np.where(mask)[0][slice_indices]
        contour_indices_global.extend(global_indices)
        contour_per_slice.append((z0, Pos_slice[slice_indices]))
        
    contour_indices_global = np.unique(contour_indices_global)
    Pos_contour = Pos[contour_indices_global]

    return Pos_contour, contour_indices_global, contour_per_slice
    


# ---------------------------------------------------------------------------
# Surface area estimation methods
# ---------------------------------------------------------------------------

def compute_surface_method_1(n_Si_atoms, n_O_atoms, n_H_atoms):
    """Approximate surface area from atom counts and van der Waals radii."""
    approx_surface = 2 * (
        np.pi * n_Si_atoms * (0.215**2)
        + np.pi * n_O_atoms * (0.158**2)
        + np.pi * n_H_atoms * (0.11**2)
    )
    return approx_surface



# Method 2: slice perimeter integration

def _order_contour_trigo(Pos_slice_contour):
    """
    Orders the contour points sequentially by following the perimeter 
    via a nearest-neighbor chain, preventing 'squiggles' caused by non-convex shapes.
    """
    xy = Pos_slice_contour[:, :2].copy()
    n_pts = len(xy)
    
    if n_pts == 0:
        return Pos_slice_contour, np.array([0, 0])
        
    center = np.mean(xy, axis=0)
    
    # Track which points have been added to our ordered path
    visited = np.zeros(n_pts, dtype=bool)
    ordered_indices = []
    
    # Start at an arbitrary point (e.g., the first one)
    current_idx = 0
    ordered_indices.append(current_idx)
    visited[current_idx] = True
    
    # Greedy walk to build a continuous loop
    for _ in range(n_pts - 1):
        current_pt = xy[current_idx]
        
        # Calculate distances from the current point to all other unvisited points
        dists = np.linalg.norm(xy - current_pt, axis=1)
        dists[visited] = np.inf  # Ignore already visited points
        
        # Move to the closest unvisited point
        next_idx = np.argmin(dists)
        ordered_indices.append(next_idx)
        visited[next_idx] = True
        current_idx = next_idx
        
    ordered_points = Pos_slice_contour[ordered_indices]
    
    return ordered_points, center

def _contour_perimeter(ordered_points):
    xy = ordered_points[:,:2]

    xy_closed = np.vstack([xy, xy[0]])
    diffs = np.diff(xy_closed, axis = 0)

    distances = np.linalg.norm(diffs, axis = 1)
    return np.sum(distances)

def compute_surface_method_2(contour_per_slice, dz = 1.0):
    total_surface = 0.0
    for z0, contour_pts in contour_per_slice:
        if len(contour_pts) < 3:
            continue
        ordered_pts, center = _order_contour_trigo(contour_pts)
        P = _contour_perimeter(ordered_pts)
        total_surface += P*dz
    return total_surface




# Method 3: circle perimeter difference calculation

def _perimeter_circle1_minus_cirlcle2(C1, R1, C2, R2):
    distance_centres = np.linalg.norm(np.array(C2) - np.array(C1))
    if distance_centres + R2 <= R1:
        return 2*np.pi*R1 
    if distance_centres >= R1 + R2:
        return 2*np.pi*R1
    cos_angle1 = np.clip((distance_centres**2 + R1**2 - R2**2)/(2*distance_centres*R1), -1, 1)
    cos_angle2 = np.clip((distance_centres**2 + R2**2 - R1**2)/(2*distance_centres*R2), -1, 1) 
    angle1 = np.arccos(cos_angle1) 
    angle2 = np.arccos(cos_angle2)

    arc_C1_outside_C2 = R1*(2*np.pi - 2*angle1) 
    arc_C2_inside_C1 = R2*(2*angle2)
     
    return arc_C1_outside_C2 + arc_C2_inside_C1

def compute_surface_method_3(Pos, list_BOX, dz = 1.0, n_points = 1000):
    z_max = list_BOX[0][2][1]
    z_values = np.arange(0, z_max, dz)
    total_surface = 0.0
    for z0 in z_values:
        mask = _slice_z(Pos, z0, dz)
        Pos_slice = Pos[mask]
        if len(Pos_slice) < 3:
            continue
        _, center1, R1 = circle(Pos_slice, z0, dz)
        if center1 is None:
            continue
        circle_pts1, _, min_dist1, theta1 = circle_min_distances(Pos_slice, center1, R1, n_points = n_points)
        center2, R2 = second_circle(circle_pts1, min_dist1, theta1)
        P = _perimeter_circle1_minus_cirlcle2(center1, R1, center2, R2)
        total_surface += P*dz

    return total_surface


# Method 4: tubular surface area estimate

def compute_surface_method_4(radius, pitch):
    approx_surface_with_cylinder = 2*2*np.pi*radius*pitch
    return approx_surface_with_cylinder




# Method 5: transformed cuboid surface estimation

def _compute_D_transfo(D_exp, pitch, width, thickness):
    W = width
    P = pitch/2/np.pi
    T = thickness


    D_est = (-T+D_exp) /2
    N = P / (P*P + D_est*D_est)**(1/2)
    b = 2*T*D_est / (W*W*N*N - T*T)
    if b > 1 :
        #The extremum point is the point in the external layer
        D_transfo = D_est
    else:
        #The extremum is in-between
        d0 = (W*W/4 - D_exp**2/4) * P**4 * (W*W-T*T)
        d2 = (P*P*D_exp**2/4 * (T*T-W*W) + P*P*T*T * (D_exp**2/4-W*W/4)) + P**4*W*W
        d4 = (D_exp**2/4 *T*T + W*W*P*P)

        delta = (d2**2 - 4*d4*d0)
        # print(delta)
        if delta > d2**2:
            D_transfo = ((-d2 + delta**(1/2))/2/d4)**(1/2)
        else:
            print("Two possiblities for the D, the higher one has been taken")
            D_transfo = ((-d2 - delta**(1/2))/2/d4)**(1/2)
            D_transfo = ((-d2 + delta**(1/2))/2/d4)**(1/2)

    return D_transfo

# Helper: quadrilateral mesh area calculation

def _quad_area(pts):
        A, B = pts[:-1, :-1], pts[1:, :-1]
        C, D = pts[1:, 1:], pts[:-1, 1:]
        t1 = 0.5*np.linalg.norm(np.cross(B-A, C-A, axis = -1), axis = -1)
        t2 = 0.5*np.linalg.norm(np.cross(C-A, D-A, axis = -1), axis = -1)
        return np.sum((t1 + t2))    

def compute_surface_method_5(Pos, diameter, width, thickness, pitch, list_BOX, n_x = 100, n_z = 100, n_y = 100, n = 10, circling = True, face = 'outer'):
    Lims_initial, Atom_types, Atom_pos = read_data("beta_quartz.data", do_scale=False,atom_style="atom")
    #Get the number of duplication needed to get the proper dimensions
    lx = Lims_initial[0][1] - Lims_initial[0][0]
    ly = Lims_initial[1][1] - Lims_initial[1][0]
    lz = Lims_initial[2][1] - Lims_initial[2][0]

    Nx = int(width // lx +1)
    Ny = int(thickness // ly +1)
    Nz = int(pitch // lz +1)

    Pos_init, _, _, _, _ = duplicate(Nx,Ny,Nz, Lims_initial, Atom_types, Atom_pos)

    xmin, xmax = np.min(Pos_init[:,0]), np.max(Pos_init[:,0])
    ymin, ymax = np.min(Pos_init[:,1]), np.max(Pos_init[:,1])
    zmin, zmax = np.min(Pos_init[:,2]), np.max(Pos_init[:,2])

    D_transfo = _compute_D_transfo(diameter, pitch, width, thickness)

    xs = np.linspace(xmin, xmax, n)
    zs = np.linspace(zmin, zmax, n)
    ys = np.linspace(ymin, ymax, n)

    XX, ZZ, YY = np.meshgrid(xs, zs, ys, indexing='ij')
    face_pts = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3)

    types_dummy = np.ones(len(face_pts), dtype=int)
    lims_dummy = np.zeros((3, 2))
    Pos_t, _ = transfo(
        face_pts,
        types_dummy,
        lims_dummy,
        {},
        D=D_transfo,
        rota=1.0,
        do_periodic=True,
        circling=True,
        do_rota_transf=False,
        params_helix=[pitch, width, thickness],
    )

    _, contour_indices_global, _ = external_surface(Pos_t, list_BOX, dz=pitch/n, n_points=n)
    pts_grid = Pos_t.reshape(n, n, n, 3)

    mask_flat = np.zeros(len(Pos_t), dtype=bool)
    mask_flat[contour_indices_global] = True
    mask_grid = mask_flat.reshape(n,n,n)

    total = _quad_area(pts_grid, mask_grid)/n
    return total


def visualize_slice_with_circles(
    Pos, 
    z0, 
    dz=1.0, 
    n_points=1000,
    point_size=40,          # Controls size of non-surface points
    surface_size=55,        # Controls size of surface points
    circle_linewidth=10.0,   # Controls width of the green/red circles
    show_labels=False,      # Set to True to display point index text
    label_spacing=100,       # Label every Nth point to avoid clutter
    highlight_method_3 = True
):
    """
    Visualizes a specific z-slice of the system with fine-grained control
    over point sizes, line widths, and text configurations.
    """
    # 1. Isolate the slice points
    mask = _slice_z(Pos, z0, dz)
    Pos_slice = Pos[mask]
    
    if len(Pos_slice) == 0:
        print(f"No atoms found in slice z = {z0} +/- {dz/2}")
        return

    xy = Pos_slice[:, :2]

    # 2. Compute the first circle (Green)
    _, center1, R1 = circle(Pos_slice, z0, dz)
    circle_pts1, closest_indices1, min_dist1, theta1 = circle_min_distances(
        Pos_slice, center1, R1, n_points=n_points
    )

    # 3. Compute the second circle (Red)
    center2, R2 = second_circle(circle_pts1, min_dist1, theta1)
    circle_pts2, closest_indices2, min_dist2, theta2 = circle_min_distances(
        Pos_slice, center2, R2, n_points=n_points
    )

    # 4. Identify Surface vs. Core points
    surface_indices = np.unique(np.concatenate([closest_indices1, closest_indices2]))
    
    is_surface = np.zeros(len(Pos_slice), dtype=bool)
    is_surface[surface_indices] = True

    surface_pts = xy[is_surface]
    remaining_pts = xy[~is_surface]

    # 5. Generate smooth circle lines for continuous plotting
    plot_theta = np.linspace(0, 2 * np.pi, 200)
    c1_line_x = center1[0] + R1 * np.cos(plot_theta)
    c1_line_y = center1[1] + R1 * np.sin(plot_theta)
    
    c2_line_x = center2[0] + R2 * np.cos(plot_theta)
    c2_line_y = center2[1] + R2 * np.sin(plot_theta)

    # 6. Plotting
    plt.figure(figsize=(9, 9))
    blue1 = (103/255.0, 179/255.0, 179/255.0) 
    blue2 = (50/255.0, 95/255.0, 255/255.0) 
    red = (192/255.0, 0/255.0, 0/255.0)
    orange = (255/255, 192/255, 0/255)
    
    # Core atoms
    plt.scatter(
        remaining_pts[:, 0], remaining_pts[:, 1], 
        color=blue2, s=70, label='Slice Points',
    )
    
    # Surface atoms (slightly larger with an outline by default)
    plt.scatter(
        surface_pts[:, 0], surface_pts[:, 1], 
        color=orange, s=100, linewidth=0.5, 
        label='Surface Points', zorder=3
    )
    
    # Bounding reference circles
    plt.plot(c1_line_x, c1_line_y, c=blue1, linewidth=circle_linewidth, label='First Circle')
    plt.plot(c2_line_x, c2_line_y, c=red, linewidth=circle_linewidth, label='Second Circle')
    
    if highlight_method_3:
        # Method 2
        contour_pts = Pos_slice[surface_indices]
        ordered_pts, _ = _order_contour_trigo(contour_pts)
        m2_xy = ordered_pts[:,:2]
        m2_closed = np.vstack([m2_xy, m2_xy[0]])
        plt.plot(m2_closed[:,0], m2_closed[:,1], c=orange, linewidth = 10.0, zorder=4, label="Slice-based Perimeter Method")
        
        
        # Method 3
        distance_centres = np.linalg.norm(np.array(center2) - np.array((center1)))
        if distance_centres + R2 <= R1:
            plt.plot(c1_line_x, c1_line_y, c='black', linewidth=10.0, label='Circle Perimeter Method')
        elif distance_centres >R1 + R2:
            plt.plot(c1_line_x, c1_line_y, c='black', linewidth=10.0, label='Circle Perimeter Method')
        else:
            center_vector_angle = np.arctan2(center2[1] - center1[1], center2[0] - center1[0])
            cos_angle1 = np.clip((distance_centres**2 + R1**2 - R2**2)/(2*distance_centres*R1), -1, 1)
            cos_angle2 = np.clip((distance_centres**2 + R2**2 - R1**2)/(2*distance_centres*R2), -1, 1) 
            angle1 = np.arccos(cos_angle1) 
            angle2 = np.arccos(cos_angle2)

            c1_pts = np.stack([c1_line_x, c1_line_y], axis = 1)
            dist_to_c2 = np.linalg.norm(c1_pts - center2, axis = 1)
            outer_c1_pts = c1_pts[dist_to_c2 >= R2 - 1e-5]
            

            c2_pts = np.stack([c2_line_x, c2_line_y], axis = 1)
            dist_to_c1 = np.linalg.norm(c2_pts - center1, axis = 1)
            inner_c2_pts = c2_pts[dist_to_c1 <= R1 + 1e-5]
            


            plt.plot(outer_c1_pts[:,0], outer_c1_pts[:,1], c='black', linewidth=10.0, alpha = 1.0, label='Circle Perimeter Method')
            plt.plot(inner_c2_pts[:,0], inner_c2_pts[:,1], c='black', linewidth=10.0, alpha = 1.0)

    # # Center markers
    # plt.scatter(center1[0], center1[1], c='green', marker='X', s=point_size * 5)
    # plt.scatter(center2[0], center2[1], c='red', marker='X', s=point_size * 5)

    # 7. Text / Label Formatting Controls
    if show_labels:
        for i, (x, y) in enumerate(xy):
            if i % label_spacing == 0:
                plt.text(
                    x, y, f"Idx: {i}\nSfc", # Multiline string to demonstrate wrapping/width controls
                    fontsize=9,
                    color='black',
                    weight='semibold',
                    ha='center',       # Horizontal alignment
                    va='bottom',       # Vertical alignment
                    wrap=True,         # Automatically wraps if given a bounding box layout
                    bbox=dict(         # Text background properties
                        boxstyle="round,pad=0.3", 
                        facecolor="white", 
                        edgecolor="gray", 
                        alpha=0.8, 
                        linewidth=30.0
                    )
                )

    # Canvas constraints
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.title(f"Helix Slice Contour Visualization", fontsize=30, pad=15)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)    
    plt.tight_layout()
    plt.show()



# ---------------------------------------------------------------------------
# Anchor molecule helpers
# ---------------------------------------------------------------------------

# XYZ loader with auto type mapping (based on element symbols in the file).
# This returns positions and inferred atom types but no bond information.
def _load_molecule_from_xyz(file, type_map = {}):
    list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_xyz(file, type_map)
    atoms = np.asarray(list_ATOMS[0])
    mol_pos = atoms[:, 2:5].astype(float)
    mol_types = []
    auto_type_map = {}
    next_id = 1
    with open(file, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].split()[0])
    for line in lines[2 : 2+n]:
        sym = line.split()[0]
        mol_type = type_map[sym]
        mol_types.append(mol_type)
        if sym not in auto_type_map:
            auto_type_map[sym]= next_id
            next_id +=1
    if type_map == {}:
        auto_type_map = {v:k for k, v in auto_type_map.items()}
    else:
        auto_type_map = {v:k for k, v in type_map.items()}
    return mol_pos, mol_types, auto_type_map



# SDF loader with bond information, and auto type mapping
def _load_molecule_from_sdf(file, type_map={}):
    """
    Read the first molecule block from a V2000 .sdf/.mol file.

    Returns
    -------
    mol_pos      : (N, 3) float array of atom positions (Ã…)
    mol_types    : list of int  â€” atom type ids (from type_map)
    bonds        : list of (bond_type, idx1, idx2)
                   bond_type follows BOND_TYPE_MAP (1-8); falls back to 1
                   if the pair is not in the dictionary.
                   idx1, idx2 are 1-based atom indices (LAMMPS convention).
    auto_type_map: dict {type_id -> element symbol}
    """
    with open(file, 'r') as f:
        lines = f.readlines()

    # Counts line is always line index 3
    counts_line = lines[3]
    n_atoms = int(counts_line[0:3])
    n_bonds  = int(counts_line[3:6])

    # Atom block 
    mol_pos   = []
    symbols   = []           # element symbol per atom
    mol_types = []

    # Build type_map incrementally if not provided
    _working_map = dict(type_map)   # symbol -> type_id
    _next_id     = max(_working_map.values(), default=0) + 1

    for i in range(n_atoms):
        line = lines[4 + i]
        x   = float(line[0:10])
        y   = float(line[10:20])
        z   = float(line[20:30])
        sym = line[31:34].strip()        # element symbol (columns 32-34)

        mol_pos.append([x, y, z])
        symbols.append(sym)

        if sym not in _working_map:
            _working_map[sym] = _next_id
            _next_id += 1
        mol_types.append(_working_map[sym])

    mol_pos = np.array(mol_pos, dtype=float)

    # auto_type_map: type_id -> symbol  (inverse of _working_map)
    auto_type_map = {v: k for k, v in _working_map.items()}

    # Bond block 
    bonds = []
    bond_start = 4 + n_atoms

    for j in range(n_bonds):
        line      = lines[bond_start + j]
        a1        = int(line[0:3])    # 1-based
        a2        = int(line[3:6])    # 1-based
        sdf_order = int(line[6:9])    # 1=single, 2=double, 3=triple, 4=aromatic

        sym1 = symbols[a1 - 1]
        sym2 = symbols[a2 - 1]
        key  = (frozenset({sym1, sym2}), sdf_order)

        bond_type = BOND_TYPE_MAP.get(key, 1)   # default to 1 if unknown
        bonds.append((bond_type, a1, a2))

    return mol_pos, mol_types, bonds, auto_type_map


# Prepare anchor molecule by removing the terminal H from N-H

def _prepare_mol(file, type_map={}):
    """
    Load anchor molecule from SDF, remove the terminal C-H (H farthest from N)
    and its bond, then adjust all remaining bond indices.

    Returns
    -------
    mol_pos_clean    : (N-1, 3)
    mol_types_clean  : list[int]
    bonds_clean      : list[(bond_type, i1, i2)]  â€“ 1-based, H-removed & re-indexed
    attach_C_idx     : int  â€“ 0-based index of the C that will bond to Si
    auto_type_map    : dict {type_id -> element symbol}
    """
    # Load molecule from SDF
    mol_pos, mol_types, bonds, auto_type_map = _load_molecule_from_sdf(file, type_map)
    symbols = np.array([auto_type_map[t] for t in mol_types])

    N_idx  = np.where(symbols == 'N')[0][0]
    N_pos  = mol_pos[N_idx]

    H_indices    = np.where(symbols == 'H')[0]
    dists_from_N = np.linalg.norm(mol_pos[H_indices] - N_pos, axis=1)
    terminal_H_idx  = H_indices[np.argmax(dists_from_N)]   # 0-based
    terminal_H_pos  = mol_pos[terminal_H_idx]

    C_indices    = np.where(symbols == 'C')[0]
    dists_from_H = np.linalg.norm(mol_pos[C_indices] - terminal_H_pos, axis=1)
    attach_C_idx = C_indices[np.argmin(dists_from_H)]      # 0-based

    # Remove the C-H bond for the deleted H 
    h1 = terminal_H_idx + 1                                 # convert to 1-based
    bonds_clean = [(bt, i1, i2) for bt, i1, i2 in bonds
                   if i1 != h1 and i2 != h1]

    # Shift every index that was above the deleted atom down by 1
    def _shift(i):
        return i - 1 if i > h1 else i

    bonds_clean = [(bt, _shift(i1), _shift(i2)) for bt, i1, i2 in bonds_clean]

    # Delete the atom from pos / types
    mol_pos_clean   = np.delete(mol_pos,   terminal_H_idx, axis=0)
    mol_types_clean = np.delete(mol_types, terminal_H_idx)

    if attach_C_idx > terminal_H_idx:
        attach_C_idx -= 1

    return mol_pos_clean, mol_types_clean, bonds_clean, attach_C_idx, auto_type_map, terminal_H_idx


# Compute outward surface normal
def _compute_outward_normal(Si_pos, tree, Pos_all, cutoff = 6.0):
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

# Align molecule to the outward normal
def _align_mol_to_normal(mol_pos, mol_types, attach_idx, outward_normal, type_map):
    mol_c = mol_pos - mol_pos[attach_idx]
    symbols = np.array([type_map[t] for t in mol_types])
    N_idx = np.where(symbols == 'N')[0][0]

    e1 = mol_c[N_idx]
    e1 = e1/np.linalg.norm(e1)
    prep = np.array([1,0,0]) if abs(e1[0]) < 0.9 else np.array([0, 1, 0])
    e2 = np.cross(e1, prep)
    e2 = e2/np.linalg.norm(e2)
    e3 = np.cross(e1, e2)

    f1 = outward_normal
    f2 = np.cross(f1, prep)
    f2 = f2/np.linalg.norm(f2)
    f3 = np.cross(f1, f2)

    aligned = np.zeros_like(mol_c)
    for i, p in enumerate(mol_c):
        c1 = np.dot(p, e1)
        c2 = np.dot(p, e2)
        c3 = np.dot(p, e3)
        aligned[i] = c1*f1 + c2*f2 + c3*f3
    
    return aligned

# ---------------------------------------------------------------------------
# Surface grafting helpers
# ---------------------------------------------------------------------------

# Extensions to graft_molecules supporting additional anchor site types

#   1. OH sites  (use_OH_sites=True)
# Atoms of type OH_types (default: {3, 4}) are artificially added OH groups.
# The anchor is placed on the Si they are bonded to, and the OH atom is
# removed from the system, exactly as is done for unsaturated Si.

#   2. O-triangle sites  (use_O_triangle_sites=True) — not sure it works correctly, needs testing
# Triplets of type-2 oxygens (each with only one bond) that lie within
# `O_triangle_radius` Å of one another form a triangle whose centre is a
# vacancy.  An artificial Si is inserted at the centroid, and the anchor
# is attached to it.

# Both features are disabled by default so the original behaviour is preserved.

from collections import defaultdict


# locate OH-site anchor positions

def find_OH_anchor_sites(
    Pos, Types,
    OH_types=(3, 4),
    Si_type=1,
    bond_cutoff=2.0,
):
    """
    Identify anchor sites corresponding to artificially added OH groups.

    For every atom whose type is in OH_types the function finds the nearest
    Si (within bond_cutoff angstrom) and records it as a candidate anchor site.
    It also returns the indices of the OH atoms so can remove them.

    """
    OH_mask    = np.isin(Types, list(OH_types))
    OH_indices = np.where(OH_mask)[0]

    Si_mask    = Types == Si_type
    Si_indices = np.where(Si_mask)[0]

    if len(OH_indices) == 0 or len(Si_indices) == 0:
        return np.empty((0, 3)), [], []

    tree_Si = cKDTree(Pos[Si_indices])

    anchor_pos = []
    valid_OH   = []
    si_for_oh  = []

    for oh_idx in OH_indices:
        dists, local_si = tree_Si.query(Pos[oh_idx], k=1)
        if dists <= bond_cutoff:
            global_si = Si_indices[local_si]
            anchor_pos.append(Pos[global_si])
            valid_OH.append(oh_idx)
            si_for_oh.append(global_si)

    if anchor_pos:
        return np.array(anchor_pos), valid_OH, si_for_oh
    return np.empty((0, 3)), [], []


# locate O-triangle anchor positions

def find_O_triangle_sites(
    Pos, Types,
    O_type=2,
    O_triangle_radius=2.0,
    bond_cutoff=2.0,
):
    """
    Find triplets of singly-bonded type-2 oxygens that form a triangle,
    and return an artificial Si position at each triangle's centroid.

    The search proceeds in three steps:
      1. Collect all type-O_type oxygens that have exactly one neighbour
         within bond_cutoff angstrom (i.e. one bond, to a surface Si).
      2. Among those, find all pairs separated by â‰¤ O_triangle_radius Ã….
      3. Report every triplet (i, j, k) in which all three pairwise distances
         satisfy that threshold.  The centroid of each such triplet is the
         position of the artificial Si to be inserted.

    Prints the number of candidate singly-bonded oxygens and triangles found.
    """
    # Step 1: singly-bonded type-2 oxygens
    O_mask    = Types == O_type
    O_indices = np.where(O_mask)[0]
    Pos_O     = Pos[O_indices]

    if len(O_indices) == 0:
        print("find_O_triangle_sites: no type-2 oxygens found.")
        return np.empty((0, 3)), []

    tree_all = cKDTree(Pos)

    single_bond = np.array([
        len(tree_all.query_ball_point(pos, r=bond_cutoff)) - 1  == 1
        for pos in Pos_O
    ])

    sb_local   = np.where(single_bond)[0]   # local indices into O_indices
    sb_global  = O_indices[sb_local]         # global indices into Pos/Types
    Pos_sb     = Pos[sb_global]

    print(f"O-triangle search: {len(sb_global)} singly-bonded type-{O_type} oxygens found.")

    if len(sb_global) < 3:
        print("  --> fewer than 3 candidates; no triangles possible.")
        return np.empty((0, 3)), []

    # Step 2: pairs within O_triangle_radius 
    tree_sb = cKDTree(Pos_sb)
    pair_set = tree_sb.query_pairs(r=O_triangle_radius)   # set of (i, j) with i < j

    # Adjacency list
    adj = defaultdict(set)
    for i, j in pair_set:
        adj[i].add(j)
        adj[j].add(i)

    # Step 3: enumerate triangles 
    triangles = []
    for i in range(len(Pos_sb)):
        for j in sorted(adj[i]):
            if j <= i:
                continue
            # k must be adjacent to both i and j, and k > j to avoid duplicates
            for k in sorted(adj[i] & adj[j]):
                if k > j:
                    triangles.append((i, j, k))

    print(f"  --> {len(triangles)} O-triangle site(s) identified.")

    if not triangles:
        return np.empty((0, 3)), []

    centroids          = []
    triangle_O_indices = []
    for (i, j, k) in triangles:
        centroid = np.mean(Pos_sb[[i, j, k]], axis=0)
        centroids.append(centroid)
        triangle_O_indices.append((sb_global[i], sb_global[j], sb_global[k]))

    return np.array(centroids), triangle_O_indices




def _fix_H_on_anchor_C(
    mol_placed, attach_C_idx, mol_types, auto_type_map,
    anchor_C_pos, outward,
    tree_surface,
    # thresholds 
    min_H_any       = 0.8,
    bond_cutoff     = 1.2,
    # rotation sweep 
    use_rotation    = True,
    n_steps         = 72,
    # outward nudge
    use_nudge       = True,
    max_nudge       = 0.5,
    n_nudge_steps   = 20,
):
    """
    Check whether H atoms directly bonded to the anchor C clash with the
    surface (closer than `min_H_any` Ã… to any surface atom).  If they do,
    attempt one or both recovery strategies:

    outward nudge (use_nudge=True)
        Translate the whole molecule along the outward normal in
        `n_nudge_steps` evenly spaced increments up to `max_nudge` Ã….
        Tried after (or instead of) rotation.

    When both are active the function first sweeps every rotation; if none
    of those clears the threshold it then tries every (rotation, nudge)
    pair, i.e. the full 2-D grid, stopping at the first success.

    """

    # identify H atoms bonded to the anchor C
    symbols  = np.array([auto_type_map[t] for t in mol_types])
    H_idx    = np.where(symbols == 'H')[0]
    bonded_H = [i for i in H_idx
                if np.linalg.norm(mol_placed[i] - anchor_C_pos) < bond_cutoff]

    if not bonded_H:
        return mol_placed, True

    # â”€â”€ orthonormal frame (identical to _align_mol_to_normal) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    f1   = outward
    prep = np.array([1,0,0]) if abs(f1[0]) < 0.9 else np.array([0,1,0])
    f2   = np.cross(f1, prep);  f2 /= np.linalg.norm(f2)
    f3   = np.cross(f1, f2)

    def _rotate(positions, angle):
        v      = positions - anchor_C_pos
        c1     = v @ f1
        c2     = v @ f2
        c3     = v @ f3
        c2_rot = c2 * np.cos(angle) - c3 * np.sin(angle)
        c3_rot = c2 * np.sin(angle) + c3 * np.cos(angle)
        return anchor_C_pos + c1[:,None]*f1 + c2_rot[:,None]*f2 + c3_rot[:,None]*f3

    def _nudge(positions, delta):
        return positions + delta * outward

    def _H_ok(positions):
        return all(
            tree_surface.query(positions[i])[0] >= min_H_any
            for i in bonded_H
        )

    if _H_ok(mol_placed):
        return mol_placed, True

    angles = (np.linspace(0, 2*np.pi, n_steps, endpoint=False)[1:]  # skip 0, already tried
              if use_rotation else [0.0])

    nudges = (np.linspace(0, max_nudge, n_nudge_steps + 1)[1:]      # skip 0, already tried
              if use_nudge else [0.0])

    if use_rotation:
        for angle in angles:
            trial = _rotate(mol_placed, angle)
            if _H_ok(trial):
                return trial, True
            
    if use_nudge and not use_rotation:
        for delta in nudges:
            trial = _nudge(mol_placed, delta)
            if _H_ok(trial):
                return trial, True

    if use_rotation and use_nudge:
        for delta in nudges:
            nudged = _nudge(mol_placed, delta)
            if _H_ok(nudged):
                return nudged, True
            for angle in angles:
                trial = _rotate(nudged, angle)
                if _H_ok(trial):
                    return trial, True

    return mol_placed, False


# Main grafting entry point


def graft_molecules(
    Pos, Types, list_BOX, Pos_contour, Types_contour,
    surface_area_nm2, mol_file, type_map, density_nm2,

    d_surface=1.7219,
    d_mol_surface=1.0,
    d_mol_mol=5.0,
    neighbor_cutoff=10.0,
    Si_type=1,
    Si_coordination=3,

    use_OH_sites=False,
    OH_types=(3, 4),
    OH_bond_cutoff=2.0,

    use_O_triangle_sites=False,
    O_triangle_type=2,
    O_triangle_radius=2.0,
    O_triangle_bond_cutoff=2.0,
):
    """
    Same as before, but now also returns `all_mol_bonds`:
    a list of (bond_type, i1, i2) with 1-based global indices into the
    final [surface | anchors] atom array.

    Includes intra-molecule bonds (from the SDF) **and** the Si-C bond
    that links each anchor to its surface site.
    """

    Lims = np.array(list_BOX[-1])
    max_molecules = int(np.floor(density_nm2 * surface_area_nm2))
    if max_molecules == 0:
        print("Target density --> 0 molecules. Returning unchanged system.")
        return Pos.copy(), Types.copy(), Lims, 0

    # _prepare_mol now returns bonds_tpl 
    mol_pos_tpl, mol_types_tpl, bonds_tpl, attach_C_idx, auto_type_map, terminal_H_idx = _prepare_mol(mol_file, type_map)
    n_atoms_mol = len(mol_types_tpl)

    OH_indices_to_remove = []

    _, Si_count_O, _ = compute_hist_neighbors(
        Pos_contour, Types_contour,
        cube=30, threshold_Si=2, threshold_O=2, threshold_H=1.3, rdf_max=5,
    )
    Si_contour_idx = np.where(Types_contour == Si_type)[0]
    insat_mask     = np.array([c == Si_coordination for c in Si_count_O])
    insat_Si_local = Si_contour_idx[insat_mask]
    Pos_insat_Si   = Pos_contour[insat_Si_local]
    print(f"Unsaturated Si sites : {len(Pos_insat_Si)}")

    candidate_positions = list(Pos_insat_Si)

    if use_OH_sites:
        oh_anchor_pos, oh_indices, _ = find_OH_anchor_sites(
            Pos, Types, OH_types=OH_types, Si_type=Si_type, bond_cutoff=OH_bond_cutoff)
        print(f"OH-site anchors : {len(oh_anchor_pos)}")
        candidate_positions.extend(oh_anchor_pos)
        OH_indices_to_remove.extend(oh_indices)
    else:
        print("OH-site anchors : disabled")

    if use_O_triangle_sites:
        tri_centroids, _ = find_O_triangle_sites(
            Pos, Types, O_type=O_triangle_type,
            O_triangle_radius=O_triangle_radius, bond_cutoff=O_triangle_bond_cutoff)
        print(f"O-triangle anchors : {len(tri_centroids)}")
        candidate_positions.extend(tri_centroids)
    else:
        print("O-triangle anchors : disabled")

    candidate_positions = (np.array(candidate_positions) if candidate_positions
                           else np.empty((0, 3)))
    n_candidates = len(candidate_positions)
    print(f"Total anchor candidates : {n_candidates}")
    print(f"Target molecules : {max_molecules} "
          f"({density_nm2:.3f} /nm^2 x {surface_area_nm2:.3f} nm^2)")

    if n_candidates == 0:
        print("No anchor sites found. Returning unchanged system.")
        return (Pos.copy(), Types.copy(), Lims, 0,
                Pos_contour.copy(), Types_contour.copy(),
                np.empty((0, 3)), np.empty((0,), dtype=int),
                [])                                          # empty bond list

    if OH_indices_to_remove:
        OH_indices_to_remove = np.unique(OH_indices_to_remove)
        keep_mask            = np.ones(len(Pos), dtype=bool)
        keep_mask[OH_indices_to_remove] = False
        Pos_clean   = Pos[keep_mask]
        Types_clean = Types[keep_mask]
        print(f"OH atoms removed : {len(OH_indices_to_remove)}")
    else:
        Pos_clean   = Pos
        Types_clean = Types

    n_surface = len(Pos_clean)    # 0-based offset for the first molecule

    all_mol_positions = []
    all_mol_types     = []
    all_mol_bonds     = []        # accumulate global bonds
    skipped   = 0
    tested    = 0
    mol_count = 0                 # counts successfully placed molecules
    tree_surface    = cKDTree(Pos_clean)
    current_mol_pos = np.empty((0, 3))

    search_offsets = np.linspace(0, 2 * np.pi, 30)
    search_offsets = sorted(search_offsets, key=abs)

    for Si_pos in candidate_positions:
        if len(all_mol_positions) >= max_molecules:
            break
        tested += 1

        outward  = _compute_outward_normal(Si_pos, tree_surface, Pos_clean,
                                           cutoff=neighbor_cutoff)
        anchor_C = Si_pos + d_surface * outward

        for angle in search_offsets:
            mol_align = _align_mol_to_normal(mol_pos_tpl, mol_types_tpl,
                                             attach_C_idx, outward, auto_type_map)
            if angle != 0:
                f1   = outward / np.linalg.norm(outward)
                prep = np.array([1, 0, 0]) if abs(f1[0]) < 0.9 else np.array([0, 1, 0])
                f2   = np.cross(f1, prep); f2 /= np.linalg.norm(f2)
                f3   = np.cross(f1, f2)
                v    = mol_align
                c1, c2, c3 = v @ f1, v @ f2, v @ f3
                c2n  = c2 * np.cos(angle) - c3 * np.sin(angle)
                c3n  = c2 * np.sin(angle) + c3 * np.cos(angle)
                mol_align = c1[:, None]*f1 + c2n[:, None]*f2 + c3n[:, None]*f3

            mol_placed = mol_align + anchor_C
            mol_placed, H_ok = _fix_H_on_anchor_C(
                mol_placed, attach_C_idx, mol_types_tpl, auto_type_map,
                anchor_C, outward, tree_surface,
                min_H_any=0.8, bond_cutoff=1.2,
                use_rotation=False, n_steps=72,
                use_nudge=True, max_nudge=0.5, n_nudge_steps=20)

            if not H_ok:
                skipped += 1
                continue

            symbols_mol = np.array([auto_type_map[t] for t in mol_types_tpl])
            free_idx    = [i for i in range(len(mol_placed))
                           if i != attach_C_idx and symbols_mol[i] != 'H']
            collision   = any(
                len(tree_surface.query_ball_point(mol_placed[i], d_mol_surface)) > 0
                for i in free_idx)

            collision_mol = False
            if len(current_mol_pos) > 0:
                tree_mol      = cKDTree(current_mol_pos)
                collision_mol = any(
                    len(tree_mol.query_ball_point(p, d_mol_mol)) > 0
                    for p in mol_placed)

            if collision or collision_mol:
                skipped += 1
                continue

            # Placement successful 
            all_mol_positions.append(mol_placed)
            all_mol_types.append(mol_types_tpl)
            current_mol_pos = np.vstack([current_mol_pos, mol_placed])

            # build global bonds for this molecule 
            # 0-based offset: surface atoms come first, then molecules in order
            offset = n_surface + mol_count * n_atoms_mol

            # Intra-molecule bonds (1-based local â†’ 1-based global)
            global_bonds = [(bt, i1 + offset, i2 + offset)
                            for bt, i1, i2 in bonds_tpl]

            # Si-C bond linking the surface Si to the anchor attachment carbon
            si_1based            = int(tree_surface.query(Si_pos)[1]) + 1
            attach_C_1based_glob = offset + attach_C_idx + 1
            global_bonds.append((SI_C_BOND_TYPE, si_1based, attach_C_1based_glob))

            all_mol_bonds.extend(global_bonds)
            mol_count += 1
            break   # stop angle search

    n_grafted = len(all_mol_positions)
    print(f"Molecules placed : {n_grafted} | skipped (collision): {n_candidates - n_grafted}")

    if all_mol_positions:
        new_pos   = np.vstack(all_mol_positions)
        new_types = np.concatenate(all_mol_types)
    else:
        new_pos   = np.empty((0, 3))
        new_types = np.empty((0,), dtype=int)

    Pos_new            = np.vstack([Pos_clean, new_pos])
    Types_new          = np.concatenate([Types_clean, new_types])
    Pos_contour_new    = np.vstack([Pos_contour, new_pos])
    Types_contour_new  = np.concatenate([Types_contour, new_types])

    print(f"Total atoms after grafting: {len(Pos_new)}")

    return (
        Pos_new, Types_new, Lims, n_grafted,
        Pos_contour_new, Types_contour_new,
        new_pos, new_types,
        all_mol_bonds,  bonds_tpl, terminal_H_idx        # new return value
    )

def compute_grafted_Si_C_distances(Pos, Types, mol_pos, mol_types, Si_type=1, C_type = 5, bond_cutoff = 2.5, dr = 0.01):
    Si_idx = np.where(Types == Si_type)[0]
    Pos_Si = Pos[Si_idx]
    tree_Si = cKDTree(Pos_Si)

    num_mol = len([types for types in mol_types if types == C_type])
    C_local = np.where(np.array(mol_types) == C_type)[0]

    distances = []
    missing = 0
    for i in C_local:
        dist, _ = tree_Si.query(mol_pos[i], k=1)
        if dist <= bond_cutoff:
            distances.append(dist)
        else:
            missing += 1
    distances = np.array(distances)

    print(f"Computed Si-C distances for {len(distances)}")   
    print(f"mean {distances.mean():.3f} Ã…, min {distances.min():.3f} Ã…, max {distances.max():.3f} Ã…")
    bins = np.arange(distances.min() - dr, distances.max() + dr, dr)
    counts, edges = np.histogram(distances, bins = bins)

    centre = 0.5*(edges[:-1] + edges[1:])
    plt.bar(centre, counts/num_mol, width=dr, align='center')
    plt.xlabel("Si-C distance (Ã…)")
    plt.ylabel("Count")
    plt.title("Distribution of Si-C distances in grafted system")
    plt.show()

    return distances


def analyze_discarded_with_plots(
    candidate_positions, mol_pos_tpl, mol_types_tpl,
    attach_C_idx, auto_type_map,
    Pos_clean, tree_surface,
    d_surface=1.7219,
    d_mol_mol=5.0,
    d_mol_surface_range=np.linspace(0.0, 5.0, 61),
    target_threshold=2.0
):
    """
    1. Filters candidates by H-fix and Mol-Mol checks.
    2. Plots the surface distance distribution (Original Function Style).
    3. Returns molecules discarded specifically by the target surface threshold.
    """
    symbols_mol = np.array([auto_type_map[t] for t in mol_types_tpl])
    free_idx = [i for i in range(len(mol_pos_tpl))
                if i != attach_C_idx and symbols_mol[i] != 'H']

    valid_d_mins = []
    discarded_mols = []
    current_accepted_pos = np.empty((0, 3))
    all_mol_positions = []
    all_mol_types     = []
    skipped           = 0
    skipped_bc_H      = 0
    skipped_bc_mol      = 0
    # Simulate the grafting loop to maintain realistic Mol-Mol competition
    for Si_pos in candidate_positions:

        if len(all_mol_positions) >= 3000:
            break

        outward = _compute_outward_normal(Si_pos, tree_surface, Pos_clean, cutoff=10.0)  
        # corrected = _validate_and_correct_normal(outward, Si_pos, tree_surface, Pos_clean, Types_clean, O_type=2, min_C_O=1.3, d_surface=d_surface)
        mol_align = _align_mol_to_normal(mol_pos_tpl, mol_types_tpl, attach_C_idx, outward, auto_type_map)
        
        anchor_C = Si_pos + d_surface * outward
        mol_placed = mol_align + anchor_C
        mol_placed, H_ok = _fix_H_on_anchor_C(mol_placed, attach_C_idx, mol_types_tpl, auto_type_map, anchor_C, outward, tree_surface,
                                                min_H_any=0.8, bond_cutoff=1.2, use_rotation=False, n_steps=72,
                                                use_nudge=True, max_nudge=0.5, n_nudge_steps=20)
        
        if not H_ok:
            skipped += 1
            skipped_bc_H +=1
            continue

        symbols_mol = np.array([auto_type_map[t] for t in mol_types_tpl])
        free_idx = [i for i in range(len(mol_placed)) if i != attach_C_idx and symbols_mol[i] != 'H']  
        collision = any(len(tree_surface.query_ball_point(mol_placed[i], d_surface)) > 0 for i in free_idx)
        # collision = any(len(tree_surface.query_ball_point(p, d_mol_surface)) > 0 for p in mol_placed)
        collision_mol = False
        if len(current_accepted_pos) > 0:
            tree_mol = cKDTree(current_accepted_pos)
            collision_mol = any(
                len(tree_mol.query_ball_point(p, d_mol_mol)) > 0
                for p in mol_placed
            )

        # if collision or collision_mol:
        #     skipped += 1
        #     continue
        if collision_mol:
            skipped +=1
            skipped_bc_mol +=1
            continue

        # D. Calculate Surface Distance
        d_min = min(tree_surface.query(mol_placed[i])[0] for i in free_idx)
        valid_d_mins.append(d_min)
        
        # Track 'discarded' subset: passed B & C, but falls below target_threshold
        if d_min < target_threshold:
            discarded_mols.append(mol_placed)
        
        # We 'accept' it for the rest of the simulation to keep the mol-mol check valid
        current_accepted_pos = np.vstack([current_accepted_pos, mol_placed])

    valid_d_mins = np.array(valid_d_mins)
    
    print("Anchors skipped because of Mol-Mol check", skipped_bc_mol)
    print("Anchors skipped because of H check", skipped_bc_H)
    print("How many candidates passed H-fix & Mol-Mol checks?", len(valid_d_mins))
    print(f"Among those, how many are below the target surface threshold of {target_threshold} Ã…?", np.sum(valid_d_mins < target_threshold))
    print(f"Mean surface distance among accepted candidates: {valid_d_mins.mean():.3f} Ã…")

    n_accepted = np.array([np.sum(valid_d_mins >= t) for t in d_mol_surface_range])
    n_total = len(valid_d_mins)

    purple          = np.array([ 96,  25, 255]) / 255
    dark_purple     = np.array([ 56,  20, 180]) / 255
    dark_dark_purple= np.array([ 34,  10, 120]) / 255

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram of all "valid" candidate distances
    bins = np.linspace(0, max(valid_d_mins.max() * 1.1, 4.0), 50)
    ax1.hist(valid_d_mins, bins=bins, color = purple,edgecolor=dark_purple)
    # ax1.axvline(d_surface, color='red', linestyle='--',
    #             label=f'Discard to the surface: {d_surface}angstrom')
    ax1.axvline(target_threshold, color=dark_dark_purple, linestyle='--',
                label=f'Discard Threshold: {target_threshold}angstrom')
    ax1.set_xlabel("Min distance to surface (angstrom)")
    ax1.set_ylabel("Number of sites")
    ax1.set_title("Distribution of distances between the molecule and the surface\n(except C_attached and H)")
    ax1.legend()

    # Right: Cumulative Acceptance Curve
    ax2.plot(d_mol_surface_range, n_accepted)
    ax2.fill_between(d_mol_surface_range, n_accepted, color= purple, alpha = 0.5)
    # ax2.axvline(d_surface, color='red', linestyle='--',
    #             label=f'Discard to the surface: {d_surface}angstrom')
    ax2.set_xlabel("d_mol_surface threshold (angstrom)")
    ax2.set_ylabel("Sites accepted")
    ax2.set_title("Accepted sites vs Surface Threshold")

    ax2r = ax2.twinx()
    ax2r.plot(d_mol_surface_range, n_accepted / n_total * 100, color=dark_dark_purple, linestyle=':')
    ax2r.set_ylabel("% Accepted", color=dark_dark_purple)
    ax2r.tick_params(axis='y', labelcolor=dark_dark_purple)
    ax2.axvline(target_threshold, color=dark_dark_purple, linestyle='--',
                label=f'Discard Threshold: {target_threshold}angstrom')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    return discarded_mols, valid_d_mins




# ---------------------------------------------------------------------------
# Chromophore grafting helpers
# ---------------------------------------------------------------------------

# Helper: split flat stacked molecule arrays into per-molecule lists

def _split_molecule_list(all_pos_stacked, all_types_stacked, n_atoms_per_mol):
    """
    graft_molecules returns np.vstack(all_mol_positions) â€” a flat (N_total x 3) array.
    This splits it back into a list of (n_atoms_per_mol x 3) arrays.
    """
    n_mols = len(all_pos_stacked) // n_atoms_per_mol
    pos_list   = [all_pos_stacked[i*n_atoms_per_mol:(i+1)*n_atoms_per_mol]   for i in range(n_mols)]
    types_list = [all_types_stacked[i*n_atoms_per_mol:(i+1)*n_atoms_per_mol] for i in range(n_mols)]
    return pos_list, types_list


# Helper: prepare chromophore by removing carboxyl OH


def _prepare_chromophore(file, type_map={}):
    # 1. Load data
    mol_pos, mol_types, bonds, auto_type_map = _load_molecule_from_sdf(file, type_map)
    symbols = np.array([auto_type_map[t] for t in mol_types])

    # 2. Build Adjacency List (1-based indices from SDF)
    adj = {i + 1: [] for i in range(len(symbols))}
    for _, a1, a2 in bonds:
        adj[a1].append(a2)
        adj[a2].append(a1)

    carboxyl_C_idx = None
    remove_OH_idx = None
    remove_H_idx = None

    # 3. Search for the Carboxyl group using connectivity
    O_indices = np.where(symbols == 'O')[0]
    for o_idx in O_indices:
        o_id = o_idx + 1  # Convert to 1-based
        neighbors = adj[o_id]
        
        # Check for bonded Hydrogen (The -OH part)
        bonded_H = [n for n in neighbors if symbols[n-1] == 'H']
        if not bonded_H:
            continue
            
        # Check for bonded Carbon
        bonded_C = [n for n in neighbors if symbols[n-1] == 'C']
        if not bonded_C:
            continue
            
        c_id = bonded_C[0]
        c_neighbors = adj[c_id]
        
        # Identification of the C in O=C-OH
        other_O = [n for n in c_neighbors if symbols[n-1] == 'O' and n != o_id]
        
        if other_O:
            carboxyl_C_idx = c_id - 1  # Back to 0-based
            remove_OH_idx = o_idx
            remove_H_idx = bonded_H[0] - 1
            break

    if carboxyl_C_idx is None:
        raise ValueError(f"Could not locate a COOH group in {file} using bond connectivity.")

    # 4. Clean up bonds and atoms (same as your previous logic)
    remove_1based = {remove_OH_idx + 1, remove_H_idx + 1}
    bonds_clean = [(bt, i1, i2) for bt, i1, i2 in bonds
                   if i1 not in remove_1based and i2 not in remove_1based]

    remove_sorted = sorted(list(remove_1based))
    def _shift_multi(i):
        return i - sum(1 for r in remove_sorted if r < i)

    bonds_clean = [(bt, _shift_multi(i1), _shift_multi(i2)) for bt, i1, i2 in bonds_clean]

    remove_indices = sorted([remove_OH_idx, remove_H_idx], reverse=True)
    mol_pos_clean = np.delete(mol_pos, remove_indices, axis=0)
    mol_types_clean = np.delete(mol_types, remove_indices)

    carboxyl_C_idx_adj = carboxyl_C_idx
    for rm in sorted(remove_indices):
        if rm < carboxyl_C_idx_adj:
            carboxyl_C_idx_adj -= 1

    return mol_pos_clean, mol_types_clean, bonds_clean, carboxyl_C_idx_adj, auto_type_map




# Chromophore axis helpers
def _align_mol_to_axis(mol_pos, attach_idx, mol_axis, target_axis, surface_normal):
    mol_c = mol_pos - mol_pos[attach_idx]

    e1 = mol_axis/np.linalg.norm(mol_axis)
    # prep = np.array([1,0,0]) if abs(e1[0]) < 0.9 else np.array([1, 0, 0])
    e2_mol = np.array([0, 0, 1])
    e3 = np.cross(e1, e2_mol)
    e3 = e3/np.linalg.norm(e3)
    e2 = np.cross(e3, e1)

    f1 = target_axis/np.linalg.norm(target_axis)
    f2 = np.cross(f1, surface_normal)
    if np.linalg.norm(f2) < 1e-6: # target_axis and surface_normal are parallel
        prep = np.array([1,0,0]) if abs(e1[0]) < 0.9 else np.array([1, 0, 0])
        f2 = np.cross(f1, prep)
    f2 = f2/np.linalg.norm(f2)
    f3 = np.cross(f1, f2)

    aligned = np.zeros_like(mol_c)
    for i, p in enumerate(mol_c):
        c1 = np.dot(p, e1)
        c2 = np.dot(p, e2)
        c3 = np.dot(p, e3)
        aligned[i] = c1*f1 + c2*f2 + c3*f3

    return aligned

def _anchor_mol_axis(mol_pos, mol_types, type_map):
    symbols = np.array([type_map[t] for t in mol_types])
    N_idx = np.where(symbols == "N")[0][0]
    C_indices = np.where(symbols == "C")[0]
    dists = np.linalg.norm(mol_pos[C_indices] - mol_pos[N_idx], axis = 1)
    base_C_idx = C_indices[np.argmax(dists)]
    return mol_pos[N_idx] - mol_pos[base_C_idx]

def _chromophore_mol_axis(mol_pos, attach_idx):
    other_mask = np.ones(len(mol_pos), dtype = bool)
    other_mask[attach_idx] = False
    centroid = np.mean(mol_pos[other_mask], axis = 0)
    return centroid - mol_pos[attach_idx]



# Anchor orientation helpers

def _anchor_N_axis(anchor_pos, anchor_types, anchor_type_map):
  
    symbols = np.array([anchor_type_map[t] for t in anchor_types])
    N_idx   = np.where(symbols == 'N')[0][0]
    N_pos   = anchor_pos[N_idx]

    axis = _anchor_mol_axis(anchor_pos, anchor_types, anchor_type_map)

    return N_pos, axis / np.linalg.norm(axis)


def compute_surface_normal_(target_pos, tree_surface, Pos, Types,search_radius=10.0):
    ids = tree_surface.query_ball_point(target_pos, r = search_radius)
    if not ids:
        raise ValueError("No surface atoms found within cutoff for normal estimation.")
    
    neighbor = tree_surface.data[ids]

    # neighbor_pos = Pos[ids]
    # neighbor_types = Types[ids]

    # Simple normal estimation: vector from target to centroid of neighbors
    centroid = np.mean(neighbor, axis=0)
    normal = target_pos - centroid
    norm_length = np.linalg.norm(normal)
    if norm_length < 1e-6:
        raise ValueError("Degenerate normal vector (target at centroid).")
    
    return normal / norm_length


def _remove_one_NH(anchor_pos, anchor_types, anchor_type_map):
    """
    Remove one H from the terminal -NH2.

    Returns
    -------
    anchor_pos_new   : array without the H
    anchor_types_new : types without the H
    remove_idx       : int  â€“ 0-based local index of the removed H  â† CHANGED
    """
    symbols   = np.array([anchor_type_map[t] for t in anchor_types])
    N_idx     = np.where(symbols == 'N')[0][0]
    N_pos     = anchor_pos[N_idx]
    H_indices = np.where(symbols == 'H')[0]

    dists_from_N = np.linalg.norm(anchor_pos[H_indices] - N_pos, axis=1)
    NH_mask      = dists_from_N < 1.2
    NH_H_idxs    = H_indices[NH_mask]

    if len(NH_H_idxs) == 0:
        raise ValueError("No H bonded to N found: anchor may already be fully substituted.")

    remove_idx       = int(NH_H_idxs[0])                    # 0-based 
    anchor_pos_new   = np.delete(anchor_pos,   remove_idx, axis=0)
    anchor_types_new = np.delete(anchor_types, remove_idx)

    return anchor_pos_new, anchor_types_new, remove_idx      

# Main chromophore grafting function


def graft_chromophores(
    Pos, Types,
    all_anchor_pos_stacked, all_anchor_types_stacked,
    n_atoms_per_anchor,
    anchor_type_map,
    chrom_file, chrom_type_map,
    anchor_bonds_tpl,           # 1-based local bonds from _prepare_mol
    d_amide_bond=1.34,
    d_chrom_surf=2.5,
    d_chrom_chrom=3.5,
    rotation_steps=30,
):
    """
    Same as before, but now:
    - accepts ``anchor_bonds_tpl`` (the 1-based local bonds returned by
      ``_prepare_mol``) so it can build anchor bonds with the deleted N-H
      stripped and indices mapped to the final global array.
    - returns ``all_bonds``: a list of (bond_type, i1, i2) with 1-based
      global indices into the final
      [Pos (surface) | updated_anchors | chromophores] atom array.
      Includes anchor intra-bonds, chromophore intra-bonds, and the
      N-C amide bond between each anchor and its chromophore.
    """

    # _prepare_chromophore now returns bonds_tpl 
    chrom_pos_t, chrom_types_t, chrom_bonds_tpl, carboxyl_C_idx, chrom_auto_map = \
        _prepare_chromophore(chrom_file, chrom_type_map)
    n_atoms_chrom      = len(chrom_types_t)
    n_atoms_anchor_mod = n_atoms_per_anchor - 1   # after one N-H is removed

    chrom_axis    = _chromophore_mol_axis(chrom_pos_t, carboxyl_C_idx)
    chrom_centered = chrom_pos_t - chrom_pos_t[carboxyl_C_idx]

    anchor_pos_list, anchor_types_list = _split_molecule_list(
        all_anchor_pos_stacked, all_anchor_types_stacked, n_atoms_per_anchor)

    n_surf    = len(Pos)          # surface-only atom count
    n_anchors = len(anchor_pos_list)

    tree_surface       = cKDTree(Pos)
    all_chrom_pos      = []
    all_chrom_types    = []
    anchor_pos_list_new   = []
    anchor_types_list_new = []
    all_bonds          = []       # CHANGED
    current_chrom_pos  = np.empty((0, 3))
    placed  = 0
    skipped = 0

    for k, (anchor_pos, anchor_types) in enumerate(zip(anchor_pos_list, anchor_types_list)):
        try:
            # _remove_one_NH now returns removed_H_local_idx 
            anchor_pos_mod, anchor_types_mod, removed_H_local_idx = \
                _remove_one_NH(anchor_pos, anchor_types, anchor_type_map)
            N_pos, anchor_axis = _anchor_N_axis(anchor_pos, anchor_types, anchor_type_map)
        except Exception:
            skipped += 1
            continue

        # build anchor bonds for this anchor 
        # 0-based offset of anchor k in the final array
        anchor_offset = n_surf + k * n_atoms_anchor_mod

        # Strip the N-H bond of the removed H, then re-index
        h1 = removed_H_local_idx + 1                        # 1-based local
        anchor_bonds_mod = [(bt, i1, i2) for bt, i1, i2 in anchor_bonds_tpl
                            if i1 != h1 and i2 != h1]

        def _shift(i):
            return i - 1 if i > h1 else i

        anchor_bonds_mod = [(bt, _shift(i1), _shift(i2))
                            for bt, i1, i2 in anchor_bonds_mod]

        # Map to global 1-based indices
        global_anchor_bonds = [(bt, i1 + anchor_offset, i2 + anchor_offset)
                               for bt, i1, i2 in anchor_bonds_mod]
        all_bonds.extend(global_anchor_bonds)

        # Chromophore placement (rotation loop, unchanged logic) 
        ideal_angle    = 0
        search_offsets = np.linspace(-np.pi, np.pi, rotation_steps)
        search_offsets = sorted(search_offsets, key=abs)
        bond_site      = N_pos + d_amide_bond * anchor_axis
        success        = False

        for offset_angle in search_offsets:
            angle          = ideal_angle + offset_angle
            surface_normal = compute_surface_normal_(N_pos, tree_surface, Pos, Types,
                                                     search_radius=10.0)
            chrom_rot      = _align_mol_to_axis(chrom_centered, carboxyl_C_idx,
                                                chrom_axis, anchor_axis, surface_normal)

            if angle != 0:
                f1   = anchor_axis / np.linalg.norm(anchor_axis)
                prep = np.array([1, 0, 0]) if abs(f1[0]) < 0.9 else np.array([0, 1, 0])
                f2   = np.cross(f1, prep); f2 /= np.linalg.norm(f2)
                f3   = np.cross(f1, f2)
                v    = chrom_rot
                c1, c2, c3 = v @ f1, v @ f2, v @ f3
                c2n  = c2 * np.cos(angle) - c3 * np.sin(angle)
                c3n  = c2 * np.sin(angle) + c3 * np.cos(angle)
                chrom_rot = c1[:, None]*f1 + c2n[:, None]*f2 + c3n[:, None]*f3

            chrom_trial = chrom_rot + bond_site

            if any(len(tree_surface.query_ball_point(p, d_chrom_surf)) > 0
                   for p in chrom_trial):
                continue

            if len(current_chrom_pos) > 0:
                tree_chrom = cKDTree(current_chrom_pos)
                if any(len(tree_chrom.query_ball_point(p, d_chrom_chrom)) > 0
                       for p in chrom_trial):
                    continue

            # Placement successful 
            all_chrom_pos.append(chrom_trial)
            all_chrom_types.append(chrom_types_t.copy())
            anchor_pos_list_new.append(anchor_pos_mod)
            anchor_types_list_new.append(anchor_types_mod)
            current_chrom_pos = np.vstack([current_chrom_pos, chrom_trial])

            # chromophore global bonds 
            chrom_offset = (n_surf
                            + n_anchors * n_atoms_anchor_mod
                            + placed * n_atoms_chrom)

            global_chrom_bonds = [(bt, i1 + chrom_offset, i2 + chrom_offset)
                                  for bt, i1, i2 in chrom_bonds_tpl]
            all_bonds.extend(global_chrom_bonds)

            #  N-C amide bond (anchor N --> chromophore carboxyl C) 
            symbols_mod         = np.array([anchor_type_map[t] for t in anchor_types_mod])
            N_idx_mod           = int(np.where(symbols_mod == 'N')[0][0])
            N_global_1based     = anchor_offset + N_idx_mod + 1
            C_global_1based     = chrom_offset  + carboxyl_C_idx + 1
            all_bonds.append((AMIDE_BOND_TYPE, N_global_1based, C_global_1based))

            placed  += 1
            success  = True
            break

        if not success:
            skipped += 1
            anchor_pos_list_new.append(anchor_pos_mod)
            anchor_types_list_new.append(anchor_types_mod)

    print(f"Chromophores placed: {placed} | skipped: {skipped}")

    updated_anchor_pos   = (np.vstack(anchor_pos_list_new)
                            if anchor_pos_list_new else np.empty((0, 3)))
    updated_anchor_types = (np.concatenate(anchor_types_list_new)
                            if anchor_types_list_new else np.empty((0,), dtype=int))
    chrom_pos_stacked    = (np.vstack(all_chrom_pos)
                            if all_chrom_pos else np.empty((0, 3)))
    chrom_types_stacked  = (np.concatenate(all_chrom_types)
                            if all_chrom_pos else np.empty((0,), dtype=int))

    Pos_final   = np.vstack([Pos, updated_anchor_pos, chrom_pos_stacked])
    Types_final = np.concatenate([Types, updated_anchor_types, chrom_types_stacked])

    return (
        Pos_final, Types_final,
        placed,
        chrom_pos_stacked, chrom_types_stacked, updated_anchor_pos, updated_anchor_types,
        all_bonds,                   # new return value
    )