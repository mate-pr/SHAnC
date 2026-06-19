"""
Analysis and system visualisation utilities for SHAnC.

This module provides bond detection, coordination analysis, radial
distribution function (RDF) computation, structural validation, and
visualization helpers for oxide and metal systems.

Features:
    - compute bond graphs from atom positions and types
    - evaluate coordination and bond statistics
    - plot histograms, RDFs, and density maps
    - create 3D visualizations with PyVista

Dependencies:
    numpy, matplotlib, pyvista, scipy
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Construction.import_libraries import *
from Construction.read_write import *


# ---------------------------------------------------------------------------
# Bond computation
# ---------------------------------------------------------------------------

def _compute_bonds(Pos, Types,
                  threshold_type1=2, threshold_type2=2, threshold_H=1.3,
                  do_count_type_3=True,
                  cation_type=1, anion_type=2):
    """
    Compute the bonds of an oxide system.

    |!| This version must only be used for small systems; it builds the full
    distance matrix and is memory-intensive for large systems.

    Atom type conventions (default SiO2):
        1 : cation  (Si)
        2 : anion   (O)
        3 : OH oxygen
        4 : H

    Parameters
    ----------
    Pos             : (N, 3) array  – atom positions
    Types           : (N,)   array  – atom type ids
    threshold_type1 : float  – cation–anion bond cutoff (Å), default 2
    threshold_type2 : float  – anion–cation bond cutoff (Å), default 2
    threshold_H     : float  – O–H bond cutoff (Å), default 1.3
    do_count_type_3 : bool   – include type-3 (OH) oxygens in anion count
    cation_type     : int    – type id of the cation, default 1
    anion_type      : int    – type id of the anion,  default 2

    Returns
    -------
    Bonds, Cat_count_An, An_count_Cat, O_count_H, H_count_O
    """
    # Select atoms by type: cations, anions (including OH oxygens when requested), and hydrogens.
    Pos_Cat = Pos[Types == cation_type]

    if do_count_type_3:
        Pos_An = Pos[(Types == anion_type) | (Types == 3)]
    else:
        Pos_An = Pos[Types == anion_type]

    Pos_H     = Pos[Types == 4]
    H_present = len(Pos_H) > 0

    # Compute all pairwise Cat-An distances and identify bonds by cutoff.
    Dist = sd.cdist(Pos_Cat, Pos_An)

    Bonds       = (Dist < threshold_type1) | (Dist < threshold_type2)
    Cat_count_An = np.sum(Bonds, axis=1)
    An_count_Cat = np.sum(Bonds, axis=0)

    if H_present:
        # Also compute O–H bonding if hydrogens are present.
        Dist_OH   = sd.cdist(Pos_H, Pos_An[:len(Pos_An)])
        Bonds_OH  = Dist_OH < threshold_H
        O_count_H = np.sum(Bonds_OH, axis=0)
        H_count_O = np.sum(Bonds_OH, axis=1)
    else:
        O_count_H, H_count_O = np.array([]), np.array([])

    return Bonds, Cat_count_An, An_count_Cat, O_count_H, H_count_O



def compute_bonds_graph(Pos,Types,cube=30,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=True,Lims=[],rdf_max=5):
    """
    compute_bonds_graph(Pos,Types,cube=30,threshold_Si=2,threshold_O=2,threshold_H=1.3,periodic=True,Lims=[])

    Compute the bonds that can be used to create a graph.
    The output is a neighbor matrix array taht can be converted to a graph using networkx nx.from_numpy_array

    Parameters
    ----------

    Pos : array
        The position of the atoms of the system
    Types: array
        The types of the atoms of the system
    cube: float, optional
        The edge of the cubes used to divide the system, 30 by default. Larger cubes are faster but are more memory intensive
    threshold_Si : float, optional
        The threshold used to consider if Si and O are bonding. 2 by default
    threshold_O : float, optional
        The threshold used to consider if O and Si are bonding. 2 by default
    threshold_H : float, optional
        The threshold used to consider if O and H are bonding. 1.3 by default
    periodic : bool, optional
        Compute as if the system was periodic, True by default
    Lims : list, optional
        The limits of the system. It is necessary for periodic computations

    Returns
    -------
        Neighbor matrix of the system

    """

    Lx,Ly,Lz = np.max(Pos,axis=0)
    lx,ly,lz = np.min(Pos,axis=0)
    if periodic:
        if len(Lims) == 0:
            print("No limits were provided, the system will be taken as NON PERIODIC. Use the Lim keyword to add limits")
            periodic = False
        else:
            lz,Lz = Lims[2]


    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    Pos_added = np.copy(Pos)
    Types_added = np.copy(Types)

    Num_Si_or = np.sum(Types==1)


    if periodic:
        #Adds atoms to the system periodically to account for the periodicity
        Dz = Lz-lz

        Pos_add_z = Pos[:,2] > (Lz - rdf_max)
        Pos_remove_z = Pos[:,2] < (lz + rdf_max)

        Pos_add_Lz = Pos[Pos_add_z] - np.array([[0,0,Dz]])
        Pos_remove_Lz = Pos[Pos_remove_z] + np.array([[0,0,Dz]])

        Pos_add = np.append(Pos_add_Lz,Pos_remove_Lz,axis=0)
        Pos_added = np.append(Pos_added,Pos_add,axis=0)

        Types_add = np.append(Types[Pos_add_z],Types[Pos_remove_z],axis=0)
        Types_added = np.append(Types_added,Types_add,axis=0)

    # print("MIN")
    # print(np.min(Pos_added[:,0]),np.max(Pos_added[:,0]))
    # print(np.min(Pos_added[:,1]),np.max(Pos_added[:,1]))
    # print(np.min(Pos_added[:,2]),np.max(Pos_added[:,2]))

    Num_Si = np.sum(Types_added==1)
    Bonds_tot = np.zeros((Num_Si,Num_Si))
    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                # print("LIMS")
                # print(((x*cube + lx - threshold_Si - 0.2)),((x+1)*cube + lx + threshold_Si + 0.2))
                # print(((y*cube + ly - threshold_Si - 0.2)),((y+1)*cube + ly + threshold_Si + 0.2))
                # print(((z*cube + lz - threshold_Si - 0.2)),((z+1)*cube + lz + threshold_Si + 0.2))

                #Slice the system inside this cube
                Pos_trunc_x = (Pos_added[:,0] > (x*cube + lx - threshold_Si - 0.2)) * (Pos_added[:,0] < ((x+1)*cube + lx + threshold_Si + 0.2))
                Pos_trunc_y = (Pos_added[:,1] > (y*cube + ly - threshold_Si - 0.2)) * (Pos_added[:,1] < ((y+1)*cube + ly + threshold_Si + 0.2))
                Pos_trunc_z = (Pos_added[:,2] > (z*cube + lz - threshold_Si - 0.2)) * (Pos_added[:,2] < ((z+1)*cube + lz + threshold_Si + 0.2))
                Pos_trunc_ind = Pos_trunc_x * Pos_trunc_y * Pos_trunc_z
                Pos_trunc = Pos_added[Pos_trunc_ind]
                Types_trunc = Types_added[Pos_trunc_ind]
                if (Types_trunc == 1).any() and (Types_trunc==2).any():

                    Bonds = _compute_bonds(Pos_trunc,Types_trunc,threshold_Si=threshold_Si,threshold_O=threshold_O,threshold_H=threshold_H)[0]
                    Bonds = Bonds.astype("float")
                    #Get Bonds Si
                    Bonds = Bonds.dot(Bonds.transpose())
                    #Set distance to 1
                    Bonds = Bonds / (Bonds + (Bonds==0)*1)
                    Pos_trunc_ind_Si = Pos_trunc_ind[Types_added==1]

                    Pos_trunc_ind_Si = np.matmul(Pos_trunc_ind_Si.reshape((len(Pos_trunc_ind_Si),1)),Pos_trunc_ind_Si.reshape((1,len(Pos_trunc_ind_Si))))

                    Bonds_tot[Pos_trunc_ind_Si] = Bonds_tot[Pos_trunc_ind_Si] + Bonds.ravel()
    Bonds_tot_or = Bonds_tot[:Num_Si_or]


    if periodic:
        Pos_add_z_Si = Pos_add_z[Types==1]
        Pos_remove_z_Si = Pos_remove_z[Types==1]
        num_add = np.sum(Pos_add_z_Si)

        Bonds_tot_or[Pos_add_z_Si] = Bonds_tot_or[Pos_add_z_Si] + Bonds_tot[Num_Si_or:Num_Si_or+num_add]
        Bonds_tot_or[Pos_remove_z_Si] = Bonds_tot_or[Pos_remove_z_Si] + Bonds_tot[Num_Si_or+num_add:]
        Bonds_tot_or = Bonds_tot_or[:,:Num_Si_or]

    Bonds_tot_or = Bonds_tot_or + Bonds_tot_or.transpose()
    Bonds_tot_or = Bonds_tot_or / (Bonds_tot_or + (Bonds_tot_or==0)*1)
    Bonds_tot_or = Bonds_tot_or - np.eye(len(Bonds_tot_or)) * Bonds_tot_or
    return Bonds_tot_or


# ---------------------------------------------------------------------------
# Neighbour histogram (scalable, periodic-aware)
# ---------------------------------------------------------------------------

def _compute_hist_neighbors(Pos, Types,
                           cube=100,
                           threshold_type1=2, threshold_type2=2, threshold_H=1.3,
                           periodic=True, Lims=[], rdf_max=5,
                           cation_type=1, anion_type=2):
    """
    Compute bond counts and RDF distances by slicing the system into sub-cubes.

    Parameters
    ----------
    Pos             : (N, 3) array
    Types           : (N,)   array
    cube            : float  – sub-cube edge length (Å), default 100
    threshold_type1 : float  – cation–anion cutoff (Å), default 2
    threshold_type2 : float  – anion–cation cutoff (Å), default 2
    threshold_H     : float  – O–H cutoff (Å), default 1.3
    periodic        : bool   – apply z-periodic boundary, default True
    Lims            : list   – [[lx,Lx],[ly,Ly],[lz,Lz]], required for PBC
    rdf_max         : float  – RDF cutoff radius (Å), default 5
    cation_type     : int    – type id of the cation, default 1
    anion_type      : int    – type id of the anion,  default 2

    Returns
    -------
    Dist_list, Cat_count_An_tot, An_count_Cat_tot
    """
    Lx, Ly, Lz = np.max(Pos, axis=0)
    lx, ly, lz = np.min(Pos, axis=0)

    # Use the system bounds from Pos unless periodic limits are explicitly provided.
    if periodic:
        if len(Lims) == 0:
            print("No limits provided – running as non-periodic.")
            periodic = False
        else:
            lz, Lz = Lims[2]

    # Divide the cell into cubic subregions for scalable neighbor counting.
    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    Pos_added   = np.copy(Pos)
    Types_added = np.copy(Types)

    Num_Cat_or = np.sum(Types == cation_type)

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

    Num_at  = len(Types)
    Num_Cat = np.sum(Types == cation_type)
    Num_An  = np.sum((Types == anion_type) | (Types == 3))
    In_trunc = np.array(
        [True] * Num_at + [False] * (len(Pos_added) - Num_at), dtype=bool
    )

    Cat_count_An_tot = np.zeros(Num_Cat)
    An_count_Cat_tot = np.zeros(Num_An)
    Dist_list = []

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):

                # Inner cube – atoms whose distances will be recorded
                mask_u = (
                    (Pos_added[:, 0] >= x*cube + lx)     & (Pos_added[:, 0] < (x+1)*cube + lx) &
                    (Pos_added[:, 1] >= y*cube + ly)     & (Pos_added[:, 1] < (y+1)*cube + ly) &
                    (Pos_added[:, 2] >= z*cube + lz)     & (Pos_added[:, 2] < (z+1)*cube + lz)
                )
                Pos_trunc_uniq   = Pos_added[mask_u]
                Types_trunc_uniq = Types_added[mask_u]

                # Padded cube – for bond counting / RDF neighbours
                mask_p = (
                    (Pos_added[:, 0] >= x*cube + lx - rdf_max) & (Pos_added[:, 0] < (x+1)*cube + lx + rdf_max) &
                    (Pos_added[:, 1] >= y*cube + ly - rdf_max) & (Pos_added[:, 1] < (y+1)*cube + ly + rdf_max) &
                    (Pos_added[:, 2] >= z*cube + lz - rdf_max) & (Pos_added[:, 2] < (z+1)*cube + lz + rdf_max)
                )
                Pos_trunc   = Pos_added[mask_p]
                Types_trunc = Types_added[mask_p]

                Ind_uniq_in_pad = (mask_u & In_trunc)[mask_p]

                if not (np.any(Types_trunc == cation_type) and np.any(Types_trunc == anion_type)):
                    continue

                _, Cat_count_An, An_count_Cat, _, _ = _compute_bonds(
                    Pos_trunc, Types_trunc,
                    threshold_type1=threshold_type1,
                    threshold_type2=threshold_type2,
                    threshold_H=threshold_H,
                    do_count_type_3=True,
                    cation_type=cation_type,
                    anion_type=anion_type,
                )

                cat_mask_pad = Types_trunc == cation_type
                an_mask_pad  = (Types_trunc == anion_type) | (Types_trunc == 3)

                Cat_count_An = Cat_count_An[Ind_uniq_in_pad[cat_mask_pad]]
                cat_idx = mask_u[:Num_at][Types == cation_type]
                Cat_count_An_tot[cat_idx] = Cat_count_An

                An_count_Cat = An_count_Cat[Ind_uniq_in_pad[an_mask_pad]]
                an_idx = mask_u[:Num_at][(Types == anion_type) | (Types == 3)]
                An_count_Cat_tot[an_idx] = An_count_Cat

                # Distances for RDF
                Pos_Cat_u = Pos_trunc_uniq[Types_trunc_uniq == cation_type]
                Pos_An_u  = Pos_trunc_uniq[
                    (Types_trunc_uniq == anion_type) | (Types_trunc_uniq == 3)
                ]
                if len(Pos_Cat_u) == 0 or len(Pos_An_u) == 0:
                    continue

                D = sd.cdist(Pos_Cat_u, Pos_An_u)
                D[D == 0] = 100
                Dist_list.append(D[D < rdf_max].ravel())

    return Dist_list, Cat_count_An_tot[:Num_Cat], An_count_Cat_tot[:Num_An]


# ---------------------------------------------------------------------------
# RDF plots
# ---------------------------------------------------------------------------

def plot_rdf_type1type2(Pos, Types,
                        threshold_type1=2, threshold_type2=2,
                        rdf_max=3.2,
                        periodic=False, Lims=[],
                        vline=1.95, density=True,
                        cation_type=1, anion_type=2,
                        title=None,
                        font_weight='normal',
                        title_font_weight='normal'):
    """
    Plot the RDF between cation (type 1) and anion (type 2) atoms.

    Parameters
    ----------
    Pos, Types      : positions and types array
    rdf_max         : float  – x-axis upper limit (Å)
    periodic        : bool   – use z-PBC
    Lims            : list   – system limits (required for PBC)
    vline           : float  – vertical reference line position (Å)
    density         : bool   – normalise to g(r); False = raw counts
    cation_type     : int    – type id of the cation, default 1
    anion_type      : int    – type id of the anion,  default 2
    title           : str    – plot title (auto-generated if None)
    font_weight     : str    – tick/label font weight
    title_font_weight : str  – title font weight
    """
    Dist_list, _, _ = _compute_hist_neighbors(
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
    for spine in ax.spines.values():
        spine.set_linewidth(5)

    counts, edges, patches = ax.hist(
        Dist_list, bins=100, range=(0, rdf_max),
        color=purple, edgecolor=dark_purple, linewidth=5,
    )

    if density:
        radius = ((np.roll(edges, 1) + edges) / 2)[1:]
        dr  = radius[1] - radius[0]
        g_r = counts / ((4 * np.pi * radius**2 * dr) * np.sum(Types == cation_type))
        for rect, val in zip(patches, g_r):
            rect.set_height(val)
        ax.set_ylabel("g(r)", color='black', weight=font_weight, fontsize=30)
        ax.set_ylim(0, 2)
    else:
        ax.set_ylabel("Number", color='black', weight=font_weight, fontsize=30)

    plot_title = title if title else f"RDF type{cation_type}–type{anion_type}"
    ax.set_title(plot_title, color='black', weight=title_font_weight, fontsize=30)
    ax.set_xlabel("Distance (Å)", color='black', weight=font_weight, fontsize=30)

    tick_locs = [k for k in range(int(rdf_max))] + [vline]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_locs, color='black', weight=font_weight, fontsize=30)
    ax.axvline(vline, color=dark_dark_purple, linestyle='--', linewidth=5,
               label=f"Bond distance {vline} Å")
    ax.legend(prop={'size': 25, 'weight': font_weight})
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.tick_params(width=5, length=5, color='black')
    ax.set_xlim(0, rdf_max)
    plt.setp(ax.get_yticklabels(), color='black', weight=font_weight, fontsize=30)
    plt.setp(ax.get_xticklabels(), color='black', weight=font_weight, fontsize=30)
    plt.tight_layout()
    plt.show()


def plot_rdf_metal(Pos, Types, type_id=1, cube=50, a_theory=4.078, bins=100, rdf_max=4.0,
                   periodic=False, Lims=[], vline=2.95,
                   font_weight='normal', title_font_weight='normal'):
    """
    Plot the metal–metal RDF.

    Parameters
    ----------
    Pos, Types    : positions and types array
    type_id       : int   – metal atom type, default 1
    cube          : float – sub-cube edge (Å), default 50
    a_theory      : float – FCC lattice parameter (Å), default 4.078 (Au)
    rdf_max       : float – RDF cutoff (Å), default 4.0
    periodic      : bool  – apply z-PBC
    Lims          : list  – system limits (required for PBC)
    vline         : float – reference bond distance marker (Å)
    font_weight, title_font_weight : str – matplotlib font weight strings
    """
    threshold  = rdf_max
    Pos_metal  = Pos
    N          = len(Pos_metal)
    if N == 0:
        print(f"No atoms with type_id={type_id} found.")
        return

    Lx, Ly, Lz = np.max(Pos_metal, axis=0)
    lx, ly, lz = np.min(Pos_metal, axis=0)

    if periodic:
        if len(Lims) == 0:
            print("No limits provided – running as non-periodic.")
            periodic = False
        else:
            lz, Lz = Lims[2]

    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    Pos_added = np.copy(Pos_metal)
    Idx_added = np.arange(N, dtype=int)

    if periodic:
        Dz       = Lz - lz
        mask_top = Pos_metal[:, 2] > (Lz - threshold)
        mask_bot = Pos_metal[:, 2] < (lz + threshold)
        Pos_added = np.vstack([
            Pos_added,
            Pos_metal[mask_top] - [0, 0, Dz],
            Pos_metal[mask_bot] + [0, 0, Dz],
        ])
        Idx_added = np.concatenate([
            Idx_added, np.where(mask_top)[0], np.where(mask_bot)[0]
        ])

    In_orig        = np.zeros(len(Pos_added), dtype=bool)
    In_orig[:N]    = True
    Dist_list      = []

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                Ind_home = (
                    (Pos_added[:, 0] >= x*cube + lx) & (Pos_added[:, 0] < (x+1)*cube + lx) &
                    (Pos_added[:, 1] >= y*cube + ly) & (Pos_added[:, 1] < (y+1)*cube + ly) &
                    (Pos_added[:, 2] >= z*cube + lz) & (Pos_added[:, 2] < (z+1)*cube + lz)
                ) & In_orig

                Ind_pad = (
                    (Pos_added[:, 0] >= x*cube + lx - threshold) & (Pos_added[:, 0] < (x+1)*cube + lx + threshold) &
                    (Pos_added[:, 1] >= y*cube + ly - threshold) & (Pos_added[:, 1] < (y+1)*cube + ly + threshold) &
                    (Pos_added[:, 2] >= z*cube + lz - threshold) & (Pos_added[:, 2] < (z+1)*cube + lz + threshold)
                )

                Pos_home  = Pos_added[Ind_home]
                Pos_pad   = Pos_added[Ind_pad]
                Idx_home  = Idx_added[Ind_home]
                Idx_pad   = Idx_added[Ind_pad]

                if len(Pos_home) < 2:
                    continue

                D = sd.cdist(Pos_home, Pos_pad)
                for local_i, gi in enumerate(Idx_home):
                    mask = (Idx_pad != gi) & (D[local_i] > 0)
                    Dist_list.append(D[local_i][mask])

    purple           = np.array([ 96,  25, 255]) / 255
    dark_purple      = np.array([ 56,  20, 180]) / 255
    dark_dark_purple = np.array([ 34,  10, 120]) / 255
    Dist_flat        = np.concatenate(Dist_list) if Dist_list else np.array([])

    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_linewidth(5)

    counts, edges, patches = ax.hist(
        Dist_flat, bins=bins, range=(0, rdf_max),
        color=purple, edgecolor=dark_purple, linewidth=5,
    )
    radius = ((np.roll(edges, 1) + edges) / 2)[1:]
    dr     = radius[1] - radius[0]
    g_r    = counts / ((4 * np.pi * radius**2 * dr) * N)
    for rect, val in zip(patches, g_r):
        rect.set_height(val)

    ax.set_title("RDF Metal–Metal", color='black', weight=title_font_weight, fontsize=30)
    ax.set_xlabel("Distance (Å)", color='black', weight=font_weight, fontsize=30)
    ax.set_ylabel("g(r)",         color='black', weight=font_weight, fontsize=30)
    tick_locs = [k for k in range(int(rdf_max))] + [vline]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_locs, color='black', weight=font_weight, fontsize=30)
    ax.axvline(vline, color=dark_dark_purple, linestyle='--', linewidth=5,
               label=f"Bond distance {vline} Å")
    ax.legend(prop={'size': 25, 'weight': font_weight})
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.tick_params(width=5, length=10, color='black')
    ax.set_xlim(0, rdf_max)
    ax.set_ylim(0, 2)
    plt.setp(ax.get_yticklabels(), color='black', weight=font_weight, fontsize=30)
    plt.setp(ax.get_xticklabels(), color='black', weight=font_weight, fontsize=30)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Metal structure validation
# ---------------------------------------------------------------------------

def check_metal_structure(Pos, Types,
                          type_id=1,
                          a_theory=4.078,
                          threshold_nn=3.2,
                          cube=30,
                          plot=True,
                          font_weight='normal',
                          title_font_weight='normal'):
    """
    Validate a transformed metal structure before MD relaxation by comparing
    nearest-neighbour distances and coordination numbers against ideal FCC
    reference values.

    Parameters
    ----------
    Pos           : (N, 3) array – atom positions (Å)
    Types         : (N,)   array – atom type ids
    type_id       : int    – metal type id, default 1
    a_theory      : float  – FCC lattice parameter (Å); Au default 4.078
    threshold_nn  : float  – neighbour cutoff (Å), default 3.2
    cube          : float  – sub-cube edge (Å), default 30
    plot          : bool   – show matplotlib figures, default True
    font_weight, title_font_weight : str – matplotlib font weight strings

    Returns
    -------
    report : dict – nn_dist_mean/std, cn_mean, cn_counts, n_too_close,
                    n_isolated, strain_mean/std
    """
    d_NN     = a_theory / np.sqrt(2)
    d_min_ok = 0.85 * d_NN
    pad      = threshold_nn + 0.2

    metal_pos = Pos[Types == type_id]
    N = len(metal_pos)
    if N == 0:
        raise ValueError(f"No atoms with type_id={type_id} found.")
    print(f"Checking {N} metal atoms (type {type_id})")

    lx, ly, lz = np.min(metal_pos, axis=0)
    Lx, Ly, Lz = np.max(metal_pos, axis=0)

    Nx = int((Lx - lx) // cube + 1)
    Ny = int((Ly - ly) // cube + 1)
    Nz = int((Lz - lz) // cube + 1)

    nn_dists    = np.full(N, np.inf)
    cn_arr      = np.zeros(N, dtype=int)
    n_too_close = 0

    for x in range(Nx):
        for y in range(Ny):
            for z in range(Nz):
                ind_uniq = (
                    (metal_pos[:, 0] >= x*cube + lx) & (metal_pos[:, 0] < (x+1)*cube + lx) &
                    (metal_pos[:, 1] >= y*cube + ly) & (metal_pos[:, 1] < (y+1)*cube + ly) &
                    (metal_pos[:, 2] >= z*cube + lz) & (metal_pos[:, 2] < (z+1)*cube + lz)
                )
                ind_pad = (
                    (metal_pos[:, 0] >= x*cube + lx - pad) & (metal_pos[:, 0] < (x+1)*cube + lx + pad) &
                    (metal_pos[:, 1] >= y*cube + ly - pad) & (metal_pos[:, 1] < (y+1)*cube + ly + pad) &
                    (metal_pos[:, 2] >= z*cube + lz - pad) & (metal_pos[:, 2] < (z+1)*cube + lz + pad)
                )

                pos_uniq = metal_pos[ind_uniq]
                pos_pad  = metal_pos[ind_pad]
                if len(pos_uniq) == 0:
                    continue

                D = sd.cdist(pos_uniq, pos_pad)
                for local_i, global_i in enumerate(np.where(ind_uniq)[0]):
                    pad_indices = np.where(ind_pad)[0]
                    col = np.where(pad_indices == global_i)[0]
                    if len(col):
                        D[local_i, col[0]] = np.inf

                nn_dists[ind_uniq]  = np.minimum(nn_dists[ind_uniq], np.min(D, axis=1))
                cn_arr[ind_uniq]   += np.sum(D < threshold_nn, axis=1)
                n_too_close        += np.sum(D < d_min_ok)

    bond_strain = (nn_dists - d_NN) / d_NN
    cn_counts   = {cn: int(np.sum(cn_arr == cn)) for cn in range(0, 14)}

    report = {
        'nn_dist_mean': float(np.mean(nn_dists)),
        'nn_dist_std':  float(np.std(nn_dists)),
        'cn_mean':      float(np.mean(cn_arr)),
        'cn_counts':    {k: v for k, v in cn_counts.items() if v > 0},
        'n_too_close':  int(n_too_close),
        'n_isolated':   int(np.sum(cn_arr == 0)),
        'strain_mean':  float(np.mean(bond_strain)),
        'strain_std':   float(np.std(bond_strain)),
    }

    if not plot:
        return report

    purple      = np.array([ 96,  25, 255]) / 255
    dark_purple = np.array([ 56,  20, 180]) / 255

    cn_vals  = sorted(report['cn_counts'].keys())
    cn_freqs = [report['cn_counts'][c] for c in cn_vals]

    fig, ax = plt.subplots()
    for spine in ax.spines.values():
        spine.set_linewidth(5)

    bars = ax.bar(cn_vals, cn_freqs, color=purple, edgecolor=dark_purple, linewidth=5)

    top_three = sorted(
        [(val / N * 100, bar, val) for bar, val in zip(bars, cn_freqs)],
        key=lambda x: x[0], reverse=True
    )[:3]
    for pct, bar, yval in top_three:
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + max(cn_freqs) * 0.02,
            f"{pct:.1f}%",
            ha='center', va='bottom',
            color=dark_purple, weight=font_weight, fontsize=25,
        )

    ax.set_title("Coordination number distribution", color='black',
                 weight=title_font_weight, fontsize=30)
    ax.set_xlabel("Coordination number", color='black', weight=font_weight, fontsize=30)
    ax.set_ylabel("Number of atoms",     color='black', weight=font_weight, fontsize=30)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.tick_params(width=5, length=5, color=dark_purple)
    ax.set_xlim(4, 14)
    ax.set_ylim(0, max(cn_freqs) * 1.15)
    plt.setp(ax.get_yticklabels(), color='black', weight=font_weight, fontsize=30)
    plt.setp(ax.get_xticklabels(), color='black', weight=font_weight, fontsize=30)
    plt.tight_layout()
    plt.show()

    return report


# ---------------------------------------------------------------------------
# Oxide analysis helpers (compute + plot)
# ---------------------------------------------------------------------------

def compute_analysis(Pos, Types, hist_Dens, hist_Si, hist_O,
                     threshold_type1=2, threshold_type2=2, threshold_H=1.3,
                     periodic=False, Lims=[], rdf_max=5, density=True,
                     cation_type=1, anion_type=2):
    """
    Compute RDF histogram and bond-count histograms (used by analyze_mult).

    Returns
    -------
    [hist_Dens, hist_Si, hist_O], Counts
    """
    Dist_list, Cat_count_An, An_count_Cat = _compute_hist_neighbors(
        Pos, Types,
        threshold_type1=threshold_type1, threshold_type2=threshold_type2,
        threshold_H=threshold_H, periodic=periodic, Lims=Lims, rdf_max=rdf_max,
        cation_type=cation_type, anion_type=anion_type,
    )
    Dist_list = [k for j in Dist_list for k in j]

    N_at       = len(Pos)
    num_inf_15 = np.sum(np.array(Dist_list) < 1.5)
    num_cat_3  = np.sum(Cat_count_An == 3)
    num_cat_5  = np.sum(Cat_count_An == 5)
    num_an_1   = np.sum(An_count_Cat == 1)
    num_an_3   = np.sum(An_count_Cat == 3)
    print(f"Bonds < 1.5 Å  : {num_inf_15}")
    print(f"3-bond cations : {num_cat_3},  5-bond cations : {num_cat_5}")
    print(f"1-bond anions  : {num_an_1},   3-bond anions  : {num_an_3}")

    hist_Dens, radius = np.histogram(Dist_list, hist_Dens[1])
    if density:
        radius    = ((np.roll(radius, 1) + radius) / 2)[1:]
        dr        = radius[1] - radius[0]
        hist_Dens = hist_Dens / (4 * np.pi * radius**2 * dr) / np.sum(Types == cation_type)

    hist_Si = np.histogram(Cat_count_An, hist_Si[1])[0]
    hist_O  = np.histogram(An_count_Cat, hist_O[1])[0]

    Counts = [np.sum(Cat_count_An), np.sum(An_count_Cat)]
    print(Counts)
    return [hist_Dens, hist_Si, hist_O], Counts


def plot_analysis(Counts_Hists, Counts, hist_Dens, hist_Si, hist_O):
    """Update histogram bar heights (used by analyze_mult)."""
    for count, rect in zip(Counts_Hists[0], hist_Dens[2].patches):
        rect.set_height(count)

    for count, rect in zip(Counts_Hists[1], hist_Si[2].patches):
        rect.set_height(count)
    hist_Si[2].set_label(f"Cat–An Bonds {Counts[0]}")

    for count, rect in zip(Counts_Hists[2], hist_O[2].patches):
        rect.set_height(count)
    hist_O[2].set_label(f"An–Cat Bonds {Counts[1]}")


def analyze_mult(list_Tstep, list_Pos, list_Types,
                 threshold_type1=2, threshold_type2=2, threshold_H=1.3,
                 rdf_max=5, periodic=False, Lims=[],
                 anim=False, save=False, vline=1.609, density=True,
                 cation_type=1, anion_type=2):
    """
    Compute and interactively display RDF and bond-count distributions for
    one or multiple timesteps.

    Parameters
    ----------
    list_Tstep    : list of int – timestep indices
    list_Pos      : list of (N,3) arrays
    list_Types    : list of (N,)  arrays
    rdf_max       : float – RDF cutoff (Å), default 5
    periodic      : bool  – z-periodic boundary
    Lims          : list  – system limits for PBC
    anim          : bool  – animate over timesteps
    save          : bool  – save last frame as analysis.svg
    vline         : float – reference vertical line on RDF plot
    density       : bool  – normalise RDF to g(r)
    cation_type   : int   – type id of the cation, default 1
    anion_type    : int   – type id of the anion,  default 2
    """
    purple      = np.array([ 96,  25, 255]) / 255
    dark_purple = np.array([ 56,  20, 180]) / 255

    fig, ax = plt.subplots()
    if save:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
        plt.rcParams.update({'font.size': 15})
        plt.rcParams['svg.fonttype'] = 'none'
    plt.axis("off")

    if (not save) and (len(list_Tstep) != 1):
        fig.subplots_adjust(bottom=0.25)
        ax_slider = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        slider = Slider(ax=ax_slider, label="Timestep",
                        valmin=list_Tstep[0], valmax=list_Tstep[-1],
                        valinit=list_Tstep[0], valfmt="%d",
                        valstep=list_Tstep)

    num_Cat = np.sum(list_Types[0] == cation_type)
    num_An  = np.sum(list_Types[0] == anion_type)

    plt.subplot(2, 1, 1)
    hist_Dens = plt.hist(np.array([]), bins=100, range=(0, rdf_max),
                         color=purple, edgecolor=dark_purple, linewidth=1, label="A")
    plt.title("RDF Cat–An", color=dark_purple)
    plt.ylabel("g(r)" if density else "Number", color=dark_purple)
    plt.xlabel("Distance (Å)", color=dark_purple)
    plt.xticks([k for k in range(int(rdf_max))] + [vline],
               [k for k in range(int(rdf_max))] + [vline], color=purple)
    plt.axvline(vline, color=dark_purple)
    plt.xlim(0, rdf_max)
    plt.ylim(0, 2 if density else num_An * 1.5)

    plt.subplot(2, 2, 3)
    hist_Si = plt.hist(np.array([]), bins=12, range=(0, 6),
                       color=purple, edgecolor=dark_purple, linewidth=1, label="A")
    plt.title("Cation bond count", color=dark_purple)
    plt.xlabel("Number of bonds", color=dark_purple)
    plt.ylabel(f"Number of type-{cation_type}", color=dark_purple)
    plt.xticks([k + 0.25 for k in range(7)], [k for k in range(7)], color=purple)
    plt.ylim(0, num_Cat * 1.2)

    plt.subplot(2, 2, 4)
    hist_O = plt.hist(np.array([]), bins=12, range=(0, 6),
                      color=purple, edgecolor=dark_purple, linewidth=1, label="A")
    plt.title("Anion bond count", color=dark_purple)
    plt.xlabel("Number of bonds", color=dark_purple)
    plt.ylabel(f"Number of type-{anion_type}", color=dark_purple)
    plt.xticks([k + 0.25 for k in range(7)], [k for k in range(7)], color=purple)
    plt.ylim(0, num_An * 1.2)

    if anim:
        plt.show(block=False)

    list_Counts_Hists, list_Counts = [], []
    for tstep in range(len(list_Tstep)):
        Hist_Counts, Counts = compute_analysis(
            list_Pos[tstep], list_Types[tstep],
            hist_Dens, hist_Si, hist_O,
            threshold_type1=threshold_type1, threshold_type2=threshold_type2,
            threshold_H=threshold_H, periodic=periodic, Lims=Lims,
            rdf_max=rdf_max, density=density,
            cation_type=cation_type, anion_type=anion_type,
        )
        if anim:
            slider.set_val(list_Tstep[tstep])
            plot_analysis(Hist_Counts, Counts, hist_Dens, hist_Si, hist_O)
            plt.pause(0.02)
        list_Counts_Hists.append(Hist_Counts)
        list_Counts.append(Counts)

    def update(val):
        index = list_Tstep.index(val)
        plot_analysis(list_Counts_Hists[index], list_Counts[index],
                      hist_Dens, hist_Si, hist_O)
        plt.draw()

    if not anim and not save and len(list_Tstep) != 1:
        slider.on_changed(update)
    if not anim and not save:
        update(list_Tstep[0])
        plt.show()

    if save:
        update(list_Tstep[-1])
        plt.tight_layout()
        plt.savefig("analysis.svg")


# ---------------------------------------------------------------------------
# Defect analysis / saving
# ---------------------------------------------------------------------------

def save_defects(file_name, Pos, Types, periodic=False, Lims=[],
                 cation_type=1, anion_type=2):
    """
    Re-map atom types to encode coordination defects and write an XYZ file.

    Normal coordination (type 1 → 1, type 2 → 2) is preserved; under- and
    over-coordinated atoms receive higher type ids for easy visualisation.
    """
    _, Cat_count_An, An_count_Cat = _compute_hist_neighbors(
        Pos, Types, cube=30, periodic=periodic, Lims=Lims, rdf_max=5,
        cation_type=cation_type, anion_type=anion_type,
    )

    # Cation re-mapping: coord 4 → type 1 (normal); others → types 5-12
    for coord, new_type in [(4, 1), (0, 5), (1, 6), (2, 7), (3, 8),
                             (5, 9), (6, 10), (7, 11), (8, 12)]:
        Types[Types == cation_type] = np.where(
            Cat_count_An == coord, new_type, Types[Types == cation_type]
        )

    # Anion re-mapping: coord 2 → type 2 (normal); others → types 13-18
    for coord, new_type in [(2, 2), (0, 13), (1, 14), (3, 15),
                             (4, 16), (5, 17), (6, 18)]:
        Types[Types == anion_type] = np.where(
            An_count_Cat == coord, new_type, Types[Types == anion_type]
        )

    write_xyz(file_name, Pos, Types)


# ---------------------------------------------------------------------------
# Surface (QuickSurf) and curvature analysis
# ---------------------------------------------------------------------------

def compute_quick_surface(Pos, grid, Lims, alpha=2, prec=20, d=10,
                           length_box=20, N_th=8):
    """
    Compute a Gaussian-density isosurface (VMD QuickSurf-style).

    This function evaluates a Gaussian density field on a 3D grid by dividing
    the space into smaller boxes and summing contributions from nearby atoms.

    ρ(r) = Σ exp(−|r − rᵢ|² / 2α²)

    The system is divided into boxes to avoid building the full distance
    matrix; multithreading is supported.

    Parameters
    ----------
    Pos        : (N, 3) array – atom positions
    grid       : pv.ImageData – grid on which to evaluate the density
    Lims       : [[Lx,lx],[Ly,ly],[Lz,lz]] – system extents
    alpha      : float – Gaussian width (Å), default 2
    prec       : float – atom-box padding beyond grid-box edge (Å), default 20
    d          : float – system extent expansion (Å), default 10
    length_box : float – box edge length (Å), default 20
    N_th       : int   – number of threads, default 8

    Returns
    -------
    cube : (M,) array – density values at each grid point
    """
    import threading as th

    x, y, z  = grid.points.T
    Pos_Grid = grid.points
    # Evaluate in parallel over z-slab ranges to accelerate the surface
    # density computation on large grids.

    Lx, lx = Lims[0]
    Ly, ly = Lims[1]
    Lz, lz = Lims[2]

    Nx_box = int((Lx - lx + 2*d) / length_box) + 1
    Ny_box = int((Ly - ly + 2*d) / length_box) + 1
    Nz_box = int((Lz - lz + 2*d) / length_box) + 1

    def evaluate_surface(cube, Nx_box, Ny_box, Nz_range):
        for z_box in range(Nz_range[0], Nz_range[1]):
            for y_box in range(Ny_box):
                for x_box in range(Nx_box):
                    Ind_Box = (
                        (x >= x_box*length_box + lx - d) & (x <= (x_box+1)*length_box + lx - d) &
                        (y >= y_box*length_box + ly - d) & (y <= (y_box+1)*length_box + ly - d) &
                        (z >= z_box*length_box + lz - d) & (z <= (z_box+1)*length_box + lz - d)
                    )
                    Pos_Box = Pos_Grid[Ind_Box]

                    Ind_trunc = (
                        (Pos[:, 0] >= x_box*length_box - prec + lx - d) & (Pos[:, 0] <= (x_box+1)*length_box + prec + lx - d) &
                        (Pos[:, 1] >= y_box*length_box - prec + ly - d) & (Pos[:, 1] <= (y_box+1)*length_box + prec + ly - d) &
                        (Pos[:, 2] >= z_box*length_box - prec + lz - d) & (Pos[:, 2] <= (z_box+1)*length_box + prec + lz - d)
                    )
                    Pos_trunc = Pos[Ind_trunc]

                    Dist = sd.cdist(Pos_Box, Pos_trunc)
                    cube[Ind_Box] = np.einsum("ij->i", np.exp(-Dist**2 / (2 * alpha**2)))

    cube = np.zeros(len(Pos_Grid))

    Nz_per_thread = [Nz_box // N_th] * N_th
    for j in range(Nz_box % N_th):
        Nz_per_thread[j] += 1

    ranges, count = [], 0
    for n in Nz_per_thread:
        ranges.append([count, count + n])
        count += n

    threads = []
    for rng in ranges:
        if rng[0] != rng[1]:
            t = th.Thread(target=evaluate_surface, args=(cube, Nx_box, Ny_box, rng))
            threads.append(t)
            t.start()
    for t in threads:
        t.join()

    return cube


def analyze_defects(Pos, Types, periodic=False, Lims=[],
                    Cycles=None, L_cycles=None,
                    d_spacing=5, isovalue=1., alpha=2., prec=20,
                    d=10, length_box=20, smoothing=1000, N_th=8,
                    cation_type=1, anion_type=2):
    """
    Visualise structural defects overlaid on a Gaussian isosurface.

    Defect analysis highlights under- and over-coordinated atoms on a
    QuickSurf-style contour and allows optional cycle drawing.
                    d_spacing=5, isovalue=1., alpha=2., prec=20,
                    d=10, length_box=20, smoothing=1000, N_th=8,
                    cation_type=1, anion_type=2):
    """
    _, Cat_count_An, An_count_Cat = _compute_hist_neighbors(
        Pos, Types, cube=30, periodic=periodic, Lims=Lims, rdf_max=5,
        cation_type=cation_type, anion_type=anion_type,
    )

    plotter = pv.Plotter()
    plotter.add_axes()

    sp = pv.Sphere(radius=0.6)
    Colors_Cat = ["white", "powderblue", "lightsteelblue", "dodgerblue",
                  "blue", "navy", "black"]
    for n_bonds in range(7):
        mask = Cat_count_An == n_bonds
        if n_bonds == 4 or not mask.any():
            continue
        data = pv.PolyData(Pos[Types == cation_type][mask])
        plotter.add_mesh(data.glyph(scale=False, geom=sp, orient=False),
                         opacity=1.0, color=Colors_Cat[n_bonds],
                         name=f"Cat_{n_bonds}")

    sp = pv.Sphere(radius=0.4)
    Colors_An = ["white", "orange", "red", "darkred", "black"]
    for n_bonds in range(5):
        mask = An_count_Cat == n_bonds
        if n_bonds == 2 or not mask.any():
            continue
        data = pv.PolyData(Pos[Types == anion_type][mask])
        plotter.add_mesh(data.glyph(scale=False, geom=sp, orient=False),
                         opacity=1.0, color=Colors_An[n_bonds],
                         name=f"An_{n_bonds}")

    Lx, Ly, Lz = np.max(Pos, axis=0) + d
    lx, ly, lz = np.min(Pos, axis=0) - d
    Nx = int(round((Lx - lx + 2*d) / d_spacing)) + 1
    Ny = int(round((Ly - ly + 2*d) / d_spacing)) + 1
    Nz = int(round((Lz - lz + 2*d) / d_spacing)) + 1

    grid    = pv.ImageData(dimensions=(Nx, Ny, Nz),
                           origin=(lx - d, ly - d, lz - d),
                           spacing=(d_spacing, d_spacing, d_spacing))
    cube    = compute_quick_surface(Pos, grid, [[Lx, lx], [Ly, ly], [Lz, lz]],
                                    alpha=alpha, prec=prec, d=d,
                                    length_box=length_box, N_th=N_th)
    contour = grid.contour(isosurfaces=(isovalue,), scalars=cube)
    smooth  = contour.smooth(n_iter=int(smoothing)) if smoothing else contour
    plotter.add_mesh(smooth, opacity=0.1, color="red", name="contour")
    plotter.show()


def analyze_density(Pos, periodic=False, Lims=[],
                    d_spacing=5, isovalue=5., alpha=2., prec=20,
                    d=10, length_box=20, smoothing=1000, N_th=8):
    """
    Render an isosurface coloured by local atom density.
    """
    rdf_max = 20
    Lx, Ly, Lz = np.max(Pos, axis=0) + d
    lx, ly, lz = np.min(Pos, axis=0) - d

    Nx = int(round((Lx - lx + 2*d) / d_spacing)) + 1
    Ny = int(round((Ly - ly + 2*d) / d_spacing)) + 1
    Nz = int(round((Lz - lz + 2*d) / d_spacing)) + 1

    grid    = pv.ImageData(dimensions=(Nx, Ny, Nz),
                           origin=(lx - d, ly - d, lz - d),
                           spacing=(d_spacing, d_spacing, d_spacing))
    cube    = compute_quick_surface(Pos, grid, [[Lx, lx], [Ly, ly], [Lz, lz]],
                                    alpha=alpha, prec=prec, d=d,
                                    length_box=length_box, N_th=N_th)
    contour = grid.contour(isosurfaces=(isovalue,), scalars=cube)
    smooth  = contour.smooth(n_iter=int(smoothing)) if smoothing else contour

    Pos_Surface = smooth.points
    Density     = np.array([
        np.sum(
            (Pos[:, 0] > px - rdf_max) & (Pos[:, 0] < px + rdf_max) &
            (Pos[:, 1] > py - rdf_max) & (Pos[:, 1] < py + rdf_max) &
            (Pos[:, 2] > pz - rdf_max) & (Pos[:, 2] < pz + rdf_max)
        )
        for px, py, pz in Pos_Surface
    ], dtype=float)

    Density /= np.max(Density)

    plotter = pv.Plotter()
    plotter.add_axes()
    plotter.add_mesh(smooth, opacity=1.0, cmap="cool", scalars=Density,
                     scalar_bar_args={"title": "Density",
                                      "title_font_size": 40,
                                      "label_font_size": 40})
    plotter.show()


# ---------------------------------------------------------------------------
# Visualisation – interactive slice viewer
# ---------------------------------------------------------------------------

def analyze_plot_syst(Pos, Types, periodic=False, Lims=[],
                      draw_limit=5, compute_limit=7,
                      Cycles=None, L_cycles=None,
                      cation_type=1, anion_type=2):
    """
    Plot slices of the system highlighting atoms with wrong coordination.
    A slider lets you navigate along the z-axis of the helix.

    Parameters
    ----------
    Pos, Types     : positions and types
    periodic       : bool  – z-periodic boundary, default False
    Lims           : list  – system limits for PBC
    draw_limit     : float – half-height of drawn slice (Å), default 5
    compute_limit  : float – half-height of computation slice (Å), default 7
    Cycles, L_cycles : optional cycle data from script_cycles
    cation_type    : int   – type id of the cation, default 1
    anion_type     : int   – type id of the anion,  default 2
    """
    def slide(value):
        if periodic:
            L = Lims[2][1]
            Pos_cut_l = (
                ((Pos[:, 2] >= value - compute_limit) & (Pos[:, 2] < value + compute_limit)) |
                ((Pos[:, 2] + L >= value - compute_limit) & (Pos[:, 2] + L < value + compute_limit)) |
                ((Pos[:, 2] - L >= value - compute_limit) & (Pos[:, 2] - L < value + compute_limit))
            )
        else:
            Pos_cut_l = (Pos[:, 2] > value - compute_limit) & (Pos[:, 2] < value + compute_limit)

        Pos_cutted_l   = Pos[Pos_cut_l]
        Types_cutted_l = Types[Pos_cut_l]
        _, Cat_count_An, An_count_Cat, O_count_H, _ = _compute_bonds(
            Pos_cutted_l, Types_cutted_l,
            cation_type=cation_type, anion_type=anion_type,
        )

        Pos_cut    = (Pos_cutted_l[:, 2] > value - draw_limit) & (Pos_cutted_l[:, 2] < value + draw_limit)
        Pos_cutted  = Pos_cutted_l[Pos_cut]
        Types_cutted = Types_cutted_l[Pos_cut]

        select_Cat = Pos_cut[Types_cutted_l == cation_type]
        select_An  = Pos_cut[((Types_cutted_l == anion_type) | (Types_cutted_l == 3))]
        Cat_count_An = Cat_count_An[select_Cat]
        An_count_Cat = An_count_Cat[select_An]
        if O_count_H.any():
            O_count_H_cut = O_count_H[select_An]
        else:
            O_count_H_cut = O_count_H

        sp = pv.Sphere(radius=0.6)
        Colors_Cat = ["white", "powderblue", "lightsteelblue", "dodgerblue",
                      "blue", "navy", "black"]
        for n_bonds in range(7):
            op = 0.05 if n_bonds == 4 else 1.0
            mask = Cat_count_An == n_bonds
            data = pv.PolyData(Pos_cutted[Types_cutted == cation_type][mask]) if mask.any() else pv.PolyData()
            plotter.add_mesh(data.glyph(scale=False, geom=sp, orient=False),
                             opacity=op, color=Colors_Cat[n_bonds],
                             name=f"Cat_{n_bonds}")

        sp = pv.Sphere(radius=0.4)
        Colors_An = ["white", "orange", "red", "darkred", "black"]
        for n_bonds in range(5):
            op = 0.05 if n_bonds == 2 else 1.0
            if O_count_H_cut.any():
                mask = (An_count_Cat + O_count_H_cut) == n_bonds
            else:
                mask = An_count_Cat == n_bonds
            data = (
                pv.PolyData(Pos_cutted[((Types_cutted == anion_type) | (Types_cutted == 3))][mask])
                if mask.any() else pv.PolyData()
            )
            plotter.add_mesh(data.glyph(scale=False, geom=sp, orient=False),
                             opacity=op, color=Colors_An[n_bonds],
                             name=f"An_{n_bonds}")

        data = pv.PolyData(Pos_cutted[Types_cutted == 4])
        plotter.add_mesh(data.glyph(scale=False, geom=pv.Sphere(radius=0.2), orient=False),
                         opacity=0.05, color="gray", name="H")

    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter()
    plotter.add_axes()
    z_min, z_max = np.min(Pos[:, 2]), np.max(Pos[:, 2])
    plotter.add_slider_widget(slide, [z_min, z_max],
                              value=(z_max + z_min) / 2,
                              title="Pos", fmt="%3.3e")
    plotter.show()


# ---------------------------------------------------------------------------
# Visualisation – atom rendering
# ---------------------------------------------------------------------------

def validate_unit_cell(Atom_pos, Atom_types, Lims,
                       expected_dist=1.613, tol=0.1,
                       type1=1, o_type=2, o_neighbours=4):
    """
    Verify that every type-1 atom has exactly o_neighbours type-2 neighbours
    at expected_dist ± tol (Å), using minimum-image periodic boundary conditions.
                       type1=1, o_type=2, o_neighbours=4):
    """                   
    L = np.array([Lims[0][1] - Lims[0][0],
                  Lims[1][1] - Lims[1][0],
                  Lims[2][1] - Lims[2][0]])

    Pos_type1 = Atom_pos[Atom_types == type1]
    Pos_o     = Atom_pos[Atom_types == o_type]

    all_ok = True
    for i, t1 in enumerate(Pos_type1):
        bonds = []
        for o in Pos_o:
            delta = o - t1
            delta -= L * np.round(delta / L)
            d = np.linalg.norm(delta)
            if abs(d - expected_dist) < tol:
                bonds.append(round(d, 4))
        if len(bonds) != o_neighbours:
            print(f"  [!] type-{type1} atom {i}: {len(bonds)} bond(s) at {bonds} "
                  f"— expected {o_neighbours} × {expected_dist} Å")
            all_ok = False

    if all_ok:
        print(f"  Unit cell OK: all type-{type1} have {o_neighbours} type-{o_type} "
              f"neighbours around {expected_dist} Å")
    return all_ok


def visualize_structure(Pos, Types, Lims=None, point_size=12, type_colors=None,
                        sphere=True, parallel_proj=True,
                        draw_bonds=False, max_bond_dist=2.0): 
    if type_colors is None:
        type_colors = {1: 'gold', 2: 'red', 3: 'blue', 4: [255, 255, 0]}

    radii = {1: 0.5, 2: 0.4, 3: 0.8, 4: 0.8}
    plotter = pv.Plotter()

    for t, color in type_colors.items():
        mask = Types == t
        if not np.any(mask):
            continue
        if sphere:
            geom = pv.Sphere(radius=radii.get(t, 0.5))
            mesh = pv.PolyData(Pos[mask].astype(float)).glyph(scale=False, geom=geom)
        else:
            mesh = pv.PolyData(Pos[mask].astype(float))
        plotter.add_mesh(mesh, color=color, lighting=False)

    if draw_bonds:
        tree = KDTree(Pos)
        for i, j in tree.query_pairs(r=max_bond_dist):
            if Types[i] != Types[j]:
                plotter.add_mesh(pv.Line(Pos[i], Pos[j]),
                                 color="gray", line_width=10, lighting=False)

    if Lims is not None:
        plotter.add_mesh(
            pv.Box(bounds=(Lims[0][0], Lims[0][1],
                           Lims[1][0], Lims[1][1],
                           Lims[2][0], Lims[2][1])),
            style="wireframe", color="black",
        )

    if parallel_proj:
        plotter.enable_parallel_projection()
    plotter.show()


def visualize_structure_cast_surface_separation(Pos, Types, Lims=None, point_size=12,
                                                type_colors=None, sphere=True,
                                                parallel_proj=True,
                                                cast_mask=None, cast=False):
    radii = {1: 0.5, 2: 0.4, 3: 0.8, 4: 0.8}
    plotter = pv.Plotter()

    if cast_mask is not None:
        split_colors = {"surface": [103, 179, 179], "cast": [192, 0, 0]}
        for flag, key in [(cast_mask, "cast"), (~cast_mask, "surface")]:
            if np.any(flag):
                geom = pv.Sphere(radius=0.5)
                mesh = (pv.PolyData(Pos[flag].astype(float)).glyph(scale=False, geom=geom)
                        if sphere else pv.PolyData(Pos[flag].astype(float)))
                plotter.add_mesh(mesh, color=split_colors[key], lighting=False)
    else:
        if type_colors is None:
            color = [192, 0, 0] if cast else [103, 179, 179]
            type_colors = {1: color, 2: color, 3: color, 4: color}
        # ← no plotter = pv.Plotter() here
        for t, color in type_colors.items():
            mask = Types == t
            if not np.any(mask):
                continue
            if sphere:
                geom = pv.Sphere(radius=radii.get(t, 0.5))
                mesh = pv.PolyData(Pos[mask].astype(float)).glyph(scale=False, geom=geom)
            else:
                mesh = pv.PolyData(Pos[mask].astype(float))
            plotter.add_mesh(mesh, color=color, lighting=False)

    if Lims is not None:
        plotter.add_mesh(
            pv.Box(bounds=(Lims[0][0], Lims[0][1],
                           Lims[1][0], Lims[1][1],
                           Lims[2][0], Lims[2][1])),
            style="wireframe",
        )

    if parallel_proj:
        plotter.enable_parallel_projection()
    plotter.show()

def visualize_close_contacts(Pos, Types, Lims=None, threshold=1.0,
                              point_size=8, type_colors=None, type=1):
    """
    Highlight same-type atom pairs closer than *threshold* Å.
    """
    if type_colors is None:
        type_colors = {1: "gold", 2: "red", 3: "blue"}

    plotter = pv.Plotter()
    for t, color in type_colors.items():
        mask = Types == t
        if np.any(mask):
            plotter.add_mesh(pv.PolyData(Pos[mask].astype(float)), color=color)

    Pos_type = Pos[Types == type].astype(float)
    if len(Pos_type) > 1:
        tree  = KDTree(Pos_type)
        pairs = np.array(list(tree.query_pairs(r=threshold)), dtype=int)
        if len(pairs):
            close_idx = np.unique(pairs.ravel())
            plotter.add_mesh(pv.PolyData(Pos_type[close_idx]), color="cyan")

            n     = len(pairs)
            pts   = np.vstack([Pos_type[pairs[:, 0]], Pos_type[pairs[:, 1]]])
            cells = np.empty((n, 3), dtype=int)
            cells[:, 0] = 2
            cells[:, 1] = np.arange(n)
            cells[:, 2] = np.arange(n) + n
            lm        = pv.PolyData()
            lm.points = pts
            lm.lines  = cells.ravel()
            plotter.add_mesh(lm, color="cyan", line_width=1)

            print(f"[close contacts] {n} type-{type}–type-{type} pair(s) with d < {threshold} Å")
            for i, j in pairs:
                d = np.linalg.norm(Pos_type[i] - Pos_type[j])
                print(f"  atom {i} and {j}: {d:.4f} Å")
        else:
            print(f"[close contacts] No type-{type}–type-{type} pair closer than {threshold} Å")

    if Lims is not None:
        plotter.add_mesh(
            pv.Box(bounds=(Lims[0][0], Lims[0][1],
                           Lims[1][0], Lims[1][1],
                           Lims[2][0], Lims[2][1])),
            style="wireframe",
        )
    plotter.show()

