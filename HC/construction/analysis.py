import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from read_write import *
import pyvista as pv
import scipy as sp
import scipy.spatial.distance as sd
import scipy.signal as sps
from script_analysis import *
from scipy.spatial import KDTree


def compute_bonds(Pos, Types,
                  threshold_type1=2, threshold_type2=2, threshold_H=1.3,
                  do_count_type_3=True,
                  cation_type=1, anion_type=2):

    Pos_Cat = Pos[Types == cation_type]

    if do_count_type_3:
        Pos_An = Pos[((Types == anion_type) | (Types == 3))]
    else:
        Pos_An = Pos[Types == anion_type]

    Pos_H = Pos[Types == 4]
    H_present = len(Pos_H) > 0

    Dist = sd.cdist(Pos_Cat, Pos_An)

    Bonds_Cat = Dist < threshold_type1
    Bonds_An  = Dist < threshold_type2
    Bonds     = Bonds_Cat | Bonds_An

    Cat_count_An = np.sum(Bonds, axis=1)
    An_count_Cat = np.sum(Bonds, axis=0)

    if H_present:
        Dist_OH  = sd.cdist(Pos_H, Pos_An[:len(Pos_An)])
        Bonds_OH = Dist_OH < threshold_H
        O_count_H = np.sum(Bonds_OH, axis=0)
        H_count_O = np.sum(Bonds_OH, axis=1)
    else:
        O_count_H, H_count_O = np.array([]), np.array([])

    return Bonds, Cat_count_An, An_count_Cat, O_count_H, H_count_O


def compute_hist_neighbors(Pos, Types,
                            cube=100,
                            threshold_type1=2, threshold_type2=2, threshold_H=1.3,
                            periodic=True, Lims=[], rdf_max=5,
                            cation_type=1, anion_type=2):

    Lx, Ly, Lz = np.max(Pos, axis=0)
    lx, ly, lz = np.min(Pos, axis=0)

    if periodic:
        if len(Lims) == 0:
            print("No limits provided, running as non-periodic.")
            periodic = False
        else:
            lz, Lz = Lims[2]

    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    Pos_added   = np.copy(Pos)
    Types_added = np.copy(Types)

    # ── use cation_type instead of hardcoded 1 ────────────────────────
    Num_Cat_or = np.sum(Types == cation_type)
    Num_An_or  = np.sum(Types == anion_type)

    if periodic:
        Pos_add_z    = Pos[:, 2] > (Lz - rdf_max)
        Pos_remove_z = Pos[:, 2] < (lz + rdf_max)

        Pos_added = np.vstack([
            Pos_added,
            Pos[Pos_add_z]    - [0, 0, Lz - lz],
            Pos[Pos_remove_z] + [0, 0, Lz - lz],
        ])
        Types_added = np.concatenate([
            Types_added,
            Types[Pos_add_z],
            Types[Pos_remove_z],
        ])

    Num_at = len(Types)
    Num_Cat = np.sum(Types == cation_type)
    Num_An  = np.sum((Types == anion_type)| (Types == 3))
    In_trunc = np.array(
        [True]*Num_at + [False]*(len(Pos_added) - Num_at), dtype=bool
    )

    Cat_count_An_tot = np.zeros(Num_Cat)
    An_count_Cat_tot = np.zeros(Num_An)
    Dist_list = []

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):

                # inner cube — atoms whose distances will be recorded
                mask_u = (
                    (Pos_added[:, 0] >= x*cube + lx)     & (Pos_added[:, 0] < (x+1)*cube + lx) &
                    (Pos_added[:, 1] >= y*cube + ly)     & (Pos_added[:, 1] < (y+1)*cube + ly) &
                    (Pos_added[:, 2] >= z*cube + lz)     & (Pos_added[:, 2] < (z+1)*cube + lz)
                )
                Pos_trunc_uniq   = Pos_added[mask_u]
                Types_trunc_uniq = Types_added[mask_u]

                # padded cube — for bond counting / RDF neighbours
                mask_p = (
                    (Pos_added[:, 0] >= x*cube + lx - rdf_max) & (Pos_added[:, 0] < (x+1)*cube + lx + rdf_max) &
                    (Pos_added[:, 1] >= y*cube + ly - rdf_max) & (Pos_added[:, 1] < (y+1)*cube + ly + rdf_max) &
                    (Pos_added[:, 2] >= z*cube + lz - rdf_max) & (Pos_added[:, 2] < (z+1)*cube + lz + rdf_max)
                )
                Pos_trunc   = Pos_added[mask_p]
                Types_trunc = Types_added[mask_p]

                Ind_uniq_in_pad = (mask_u & In_trunc)[mask_p]

                has_cat = np.any(Types_trunc == cation_type)
                has_an  = np.any(Types_trunc == anion_type)
                if not (has_cat and has_an):
                    continue

                # bond counts
                _, Cat_count_An, An_count_Cat, _, _ = compute_bonds(
                    Pos_trunc, Types_trunc,
                    threshold_type1=threshold_type1,
                    threshold_type2=threshold_type2,
                    threshold_H=threshold_H,
                    do_count_type_3=True,
                    cation_type=cation_type,
                    anion_type=anion_type,
                )

                cat_mask_pad = Types_trunc == cation_type
                an_mask_pad  = (Types_trunc == anion_type)| (Types_trunc == 3)

                Cat_count_An = Cat_count_An[Ind_uniq_in_pad[cat_mask_pad]]
                cat_idx = mask_u[:Num_at][Types == cation_type]
                Cat_count_An_tot[cat_idx] = Cat_count_An

                An_count_Cat = An_count_Cat[Ind_uniq_in_pad[an_mask_pad]]
                an_idx = mask_u[:Num_at][(Types == anion_type)| (Types == 3)]
                An_count_Cat_tot[an_idx] = An_count_Cat

                # distances for RDF
                Pos_Cat_u = Pos_trunc_uniq[Types_trunc_uniq == cation_type]
                Pos_An_u  = Pos_trunc_uniq[
                    (Types_trunc_uniq == anion_type) | (Types_trunc_uniq == 3)
                ]

                if len(Pos_Cat_u) == 0 or len(Pos_An_u) == 0:
                    continue

                D = sd.cdist(Pos_Cat_u, Pos_An_u)
                D[D == 0] = 100          # mask self-pairs
                D = D[D < rdf_max].ravel()
                Dist_list.append(D)

    return Dist_list, Cat_count_An_tot[:Num_Cat], An_count_Cat_tot[:Num_An]



def plot_rdf_sio(Pos, Types,
                 threshold_type1=2, threshold_type2=2,
                 rdf_max=3.2,
                 periodic=False, Lims=[],
                 vline=1.609, density=True,
                 cation_type=1, anion_type=2,
                 title=None):

    Dist_list, _, _ = compute_hist_neighbors(
        Pos, Types,
        threshold_type1=threshold_type1, threshold_type2=threshold_type2,
        periodic=periodic, Lims=Lims, rdf_max=rdf_max,
        cation_type=cation_type, anion_type=anion_type,
    )
    Dist_list = [k for j in Dist_list for k in j]

    purple           = np.array([ 96,  25, 255]) / 255
    dark_purple      = np.array([ 56,  20, 180]) / 255
    dark_dark_purple = np.array([ 34,  10, 120]) / 255

    fig, ax = plt.subplots()
    counts, edges, patches = ax.hist(
        Dist_list, bins=100, range=(0, rdf_max),
        color=purple, edgecolor=dark_purple, linewidth=1
    )

    if density:
        radius = ((np.roll(edges, 1) + edges) / 2)[1:]
        dr  = radius[1] - radius[0]
        g_r = counts / ((4*np.pi*radius**2*dr) * np.sum(Types == cation_type))
        for rect, val in zip(patches, g_r):
            rect.set_height(val)
        ax.set_ylabel("g(r)", color=dark_purple)
        ax.set_ylim(0, 2)
    else:
        ax.set_ylabel("Number", color=dark_purple)

    plot_title = title if title else f"RDF type{cation_type}-type{anion_type}"
    ax.set_title(plot_title,   color=dark_purple)
    ax.set_xlabel("Distance (Å)", color=dark_purple)
    ax.set_xticks([k for k in range(int(rdf_max))] + [vline])
    ax.set_xticklabels([k for k in range(int(rdf_max))] + [vline], color=purple)
    ax.tick_params(colors=purple)
    ax.axvline(vline, color=dark_dark_purple)
    ax.set_xlim(0, rdf_max)
    plt.tight_layout()
    plt.show()


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


# def plot_rdf_sio(Pos, Types, threshold_Si = 2, threshold_O = 2, rdf_max = 3.2, periodic = False, Lims = [], vline = 1.609, density = True):
#     Dist_list, _, _ = compute_hist_neighbors(Pos, Types, threshold_Si=threshold_Si, threshold_O=threshold_O, periodic=periodic, Lims = Lims, rdf_max=rdf_max)
#     Dist_list = [k for j in Dist_list for k in j]

#     purple      = np.array([96,  25, 255]) / 255
#     dark_purple = np.array([56,  20, 180]) / 255
#     dark_dark_purple = np.array([34,  10, 120]) / 255

#     fig, ax = plt.subplots()
#     counts, edges, patches =ax.hist(Dist_list, bins=100, range=(0, rdf_max), color = purple, edgecolor = dark_purple, linewidth = 1)

#     if density:
#         radius = ((np.roll(edges, 1)+ edges)/2)[1:]
#         dr = radius[1] - radius[0]
#         # V = (Lims[0][1]-Lims[0][0])*(Lims[1][1]-Lims[1][0])*(Lims[2][1]-Lims[2][0])
#         g_r = counts/((4*np.pi*radius**2*dr) * (np.sum(Types == 1))) 
#         for rect, val in zip(patches, g_r):
#             rect.set_height(val)
#         ax.set_ylabel("g(r)", color = dark_purple)
#         ax.set_ylim(0, 2)
#     else:
#         ax.set_ylabel("Number", color = dark_purple)
    
    
#     ax.set_title("RDF type1-O",
#                  color=dark_purple)
#     ax.set_xlabel("Distance (A)", color=dark_purple)
#     ax.set_ylabel("Number of pairs", color=dark_purple)
#     ax.set_xticks([k for k in range(int(rdf_max))]+[vline])
#     ax.set_xticklabels([k for k in range(int(rdf_max))]+[vline], color = purple)

#     ax.tick_params(colors=purple)
#     ax.axvline(vline, color=dark_dark_purple)
#     ax.set_xlim(0, rdf_max)
#     plt.tight_layout()
#     plt.show()



def visualize_cuboid(Pos, Types, Lims = None, point_size = 8, type_colors = None, short_dist_threshold = 0.0):
    if type_colors is None:
       type_colors = {1: "gold", 2: "red", 3: "blue", 4:"green"}
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

            print(f"[close contacts] {n} Au-Au pair(s) with d < {threshold} angstrom")
            for i, j in pairs:
                d = np.linalg.norm(Pos_au[i] - Pos_au[j])
                print(f"  atom {i} and {j}  :  {d:.4f} angstrom")
        else:
            print(f"[close contacts] No Au-Au pair closer than {threshold} angstrom")

    # ---- optional box ----
    if Lims is not None:
        box = pv.Bobounds=(Lims[0][0], Lims[0][1],
                             Lims[1][0], Lims[1][1],
                             Lims[2][0], Lims[2][1])
        plotter.add_mesh(box, style="wireframe")

    plotter.show()



def visualize_si_o_contacts(
    Pos, Types, Lims=None,
    ok_min=1.55, ok_max=1.65,
    search_radius=2.5,
    point_size=8,
    type_colors=None,
    si_type=1, o_type=2,
):
    """
    Render all atoms and draw Si-O bonds colour-coded by distance:
      - lime  : OK bonds in [ok_min, ok_max] Å
      - cyan  : BAD bonds outside that range (highlighted atoms too)

    Parameters
    ----------
    ok_min, ok_max : acceptable Si-O bond length window (Å)
    search_radius  : upper distance cutoff for neighbour search (Å)
    si_type, o_type: LAMMPS type ids for Si and O
    """
    if type_colors is None:
        type_colors = {1: "gold", 2: "red", 3: "blue"}

    plotter = pv.Plotter()

    # --- render all atoms ---
    for t, color in type_colors.items():
        mask = Types == t
        if not np.any(mask):
            continue
        plotter.add_mesh(
            pv.PolyData(Pos[mask].astype(float)),
            color=color, point_size=point_size,
            render_points_as_spheres=True,
        )

    # --- find Si-O pairs within search_radius ---
    Pos_si  = Pos[Types == si_type].astype(float)
    Pos_o   = Pos[Types == o_type ].astype(float)
    idx_si  = np.where(Types == si_type)[0]
    idx_o   = np.where(Types == o_type )[0]

    if len(Pos_si) == 0 or len(Pos_o) == 0:
        print("No Si or O atoms found.")
        plotter.show()
        return

    tree_o  = KDTree(Pos_o)
    ok_pairs, bad_pairs = [], []

    for si_pos, gi in zip(Pos_si, idx_si):
        for lj in tree_o.query_ball_point(si_pos, r=search_radius):
            d = np.linalg.norm(si_pos - Pos_o[lj])
            entry = (gi, idx_o[lj], d, si_pos.copy(), Pos_o[lj].copy())
            if ok_min <= d <= ok_max:
                ok_pairs.append(entry)
            else:
                bad_pairs.append(entry)

    # --- helper: draw bond lines ---
    def _draw_bonds(pairs, color, lw):
        if not pairs:
            return
        n   = len(pairs)
        pts = np.vstack([np.array([p[3] for p in pairs]),
                         np.array([p[4] for p in pairs])])
        cells       = np.empty((n, 3), dtype=int)
        cells[:, 0] = 2
        cells[:, 1] = np.arange(n)
        cells[:, 2] = np.arange(n) + n
        lm        = pv.PolyData()
        lm.points = pts
        lm.lines  = cells.ravel()
        plotter.add_mesh(lm, color=color, line_width=lw)

    _draw_bonds(ok_pairs,  "lime", 1)
    _draw_bonds(bad_pairs, "cyan", 3)

    # --- highlight bad-contact atoms ---
    if bad_pairs:
        bad_idx = np.unique(
            [p[0] for p in bad_pairs] + [p[1] for p in bad_pairs]
        )
        plotter.add_mesh(
            pv.PolyData(Pos[bad_idx].astype(float)),
            color="cyan", point_size=point_size * 2,
            render_points_as_spheres=True,
        )
        print(f"\n[Si-O contacts] {len(ok_pairs)} OK  bonds in [{ok_min}, {ok_max}] Å")
        print(f"[Si-O contacts] {len(bad_pairs)} BAD bonds outside that range:")
        for gi, gj, d, *_ in sorted(bad_pairs, key=lambda x: x[2]):
            tag = "too short" if d < ok_min else "too long"
            print(f"  Si[{gi}] – O[{gj}] : {d:.4f} Å  ({tag})")
    else:
        print(f"[Si-O contacts] All {len(ok_pairs)} bonds OK ✓")

    # --- optional box ---
    if Lims is not None:
        plotter.add_mesh(
            pv.Box(bounds=(Lims[0][0], Lims[0][1],
                           Lims[1][0], Lims[1][1],
                           Lims[2][0], Lims[2][1])),
            style="wireframe", color="white",
        )

    plotter.show()

### 



def validate_unit_cell(Atom_pos, Atom_types, Lims, expected_dist=1.613, tol=0.1, type1=1, o_type=2, o_neighbours = 4):
    """Check that every type 1 has exactly o_neighbours O neighbours at expected_dist±tol, using PBC."""
    Lx = Lims[0][1] - Lims[0][0]
    Ly = Lims[1][1] - Lims[1][0]
    Lz = Lims[2][1] - Lims[2][0]
    L  = np.array([Lx, Ly, Lz])

    Pos_type1 = Atom_pos[Atom_types == type1]
    Pos_o  = Atom_pos[Atom_types == o_type]

    all_ok = True
    for i, t1 in enumerate(Pos_type1):
        bonds = []
        for o in Pos_o:
            delta = o - t1
            delta -= L * np.round(delta / L)   # minimum image
            d = np.linalg.norm(delta)
            if abs(d - expected_dist) < tol:
                bonds.append(round(d, 4))
        if len(bonds) != o_neighbours:
            print(f"  [!] type 1 {i}: {len(bonds)} bond(s) found at {bonds} "
                  f"— expected {o_neighbours} x {expected_dist} angstrom")
            all_ok = False
    if all_ok:
        print(f"  Unit cell OK: all Si have 4 O neighbours around {expected_dist} Å")
    return all_ok



