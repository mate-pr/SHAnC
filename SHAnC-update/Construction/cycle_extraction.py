"""
Cycle extraction and visualization for SHAnC.

This module provides tools to detect Si–O ring cycles from bond graphs,
clean redundant cycles, save / read cycle descriptions, and visualize
cycle populations and 3D cycle geometries.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Construction.import_libraries import *
from Construction.analysis import *


# ---------------------------------------------------------------------------
# Cycle plotting and filtering helpers
# ---------------------------------------------------------------------------

def plot_cycles(L_cycles, cube=0, s=3.5):
    """Plot cycle length statistics.

    Parameters
    ----------
    L_cycles : list[int]
        List of cycle lengths.
    cube : float, optional
        Length scale used to compute the maximum reliable cycle size.
    s : float, optional
        Geometry parameter used in the precision-limit estimate.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    plt.rcParams.update({'font.size': 15})
    plt.rcParams['svg.fonttype'] = 'none'

    max_graph = np.max(L_cycles) + 1
    maxx = max_graph

    if cube != 0:
        # Maximum identifiable cycle size based on the current discretisation.
        max_cycle = int(np.pi / np.arcsin(s / cube))
        maxx = max(max_graph, max_cycle)
        ax.axvline(max_cycle, linestyle="dashed", color="blue", linewidth=3,
                   label="Precision Limit")

    purple = np.array([96, 25, 255]) / 255
    dark_purple = np.array([56, 20, 180]) / 255

    ax.hist(L_cycles, bins=maxx - 1, range=(1, maxx),
            color=purple, edgecolor=dark_purple, linewidth=1)
    ax.set_xticks([k + 0.5 for k in range(1, maxx)])
    ax.set_xticklabels([k for k in range(1, maxx)], color=purple)
    ax.set_yticklabels(ax.get_yticks(), color=purple)
    ax.set_ylabel("Number of cycles", color=dark_purple)
    ax.set_xlabel("Length of cycles", color=dark_purple)
    ax.legend()
    fig.tight_layout()
    fig.savefig("Cycles.png")


# ---------------------------------------------------------------------------
# Cycle cleaning and detection
# ---------------------------------------------------------------------------

def clean_cycles(Cycles):
    """Reduce a set of raw cycles to a minimal independent basis."""
    # Track which cycles remain active and the set representation for each.
    Cycles_S = [True] * len(Cycles)
    Cycles_set = [set(cycle) for cycle in Cycles]

    def min_basis(Sub_Cycles):
        """Build a minimal cycle basis from a set of connected subcycles."""
        G = nx.Graph()
        K = [int(d) for c in Sub_Cycles for d in c]
        G.add_nodes_from(K)

        # Add each edge only once, even when it appears in multiple cycle paths.
        L = np.zeros((np.max(K) + 1, np.max(K) + 1), dtype=bool)
        D = []
        for c in Sub_Cycles:
            cy2 = np.roll(c, 1)
            for c1, c2 in zip(c, cy2):
                if not L[c1, c2]:
                    L[c1, c2] = True
                    L[c2, c1] = True
                    D.append([int(c1), int(c2)])
        G.add_edges_from(D)

        Min_Basis = nx.minimum_cycle_basis(G)
        Basis = [list(k) for k in Min_Basis]
        Basis_sorted = [set(k) for k in Min_Basis]
        return Basis, Basis_sorted

    def _find_common(Cycles, cycle, Cycles_S):
        """Return all active cycles that share an edge with the current cycle."""
        Common_edges = []
        cycle_rolled = np.roll(cycle, 1)

        for ind_a, ind_b in zip(cycle, cycle_rolled):
            for cycle_2, index_cycle_2 in zip(Cycles, range(len(Cycles))):
                if not Cycles_S[index_cycle_2]:
                    continue
                if ind_a in cycle_2:
                    index_2 = cycle_2.index(ind_a)
                    if index_2 == len(cycle_2) - 1:
                        if cycle_2[-2] == ind_b:
                            Common_edges.append(cycle_2)
                            break
                    elif ind_b == cycle_2[index_2 + 1] or ind_b == cycle_2[index_2 - 1]:
                        Common_edges.append(cycle_2)
                        break
        return Common_edges

    Cycles_index = list(range(len(Cycles)))
    for cycle, index_cycle in zip(Cycles, Cycles_index):
        Sub_Cycles = _find_common(Cycles, cycle, Cycles_S)
        Basis, Basis_set = min_basis(Sub_Cycles)

        cycle_set = set(cycle)
        if cycle_set not in Basis_set:
            Cycles_S[index_cycle] = False
            for cycle_basis in Basis:
                cycle_basis_set = set(cycle_basis)
                if cycle_basis_set not in Cycles_set:
                    Cycles.append(cycle_basis)
                    Cycles_S.append(True)
                    Cycles_index.append(max(Cycles_index) + 1)
                    Cycles_set.append(cycle_basis_set)

    return [cycle for cycle, kept in zip(Cycles, Cycles_S) if kept]

def find_cycles(Bonds):
    """Extract simple cycles from a bond adjacency matrix."""
    G = nx.from_numpy_array(Bonds)
    start = time.time()
    Cycles = []
    Cycles_set = []
    L_cycles = []

    for node in G.nodes:
        neighbors = list(nx.neighbors(G, node))
        for n1 in neighbors:
            for n2 in neighbors:
                if n1!=n2:
                    G.remove_edges_from([(node, n1), (node, n2)])
                    try:
                        APSP = nx.bidirectional_shortest_path(G, n1, n2)
                        cycle = [node] + APSP
                        cycle_set = set(cycle)
                        if cycle_set not in Cycles_set:
                            Cycles.append(cycle)
                            Cycles_set.append(cycle_set)
                            L_cycles.append(len(cycle))
                    except nx.NetworkXNoPath:
                        pass
                    finally:
                        G.add_edges_from([(node, n1), (node, n2)])

    print(time.time() - start)
    return Cycles, L_cycles


# ---------------------------------------------------------------------------
# Cycle visualization and output
# ---------------------------------------------------------------------------

def visualize_cycles(Pos, Types, Cycles):
    """Render a 3D visualization of the detected cycles on the Si sublattice."""
    start = time.time()
    Bonds = compute_bonds_graph(Pos, Types, cube=50, periodic=False, Lims=list_BOX[-1])
    print(time.time() - start)

    colors = ["", "", "", "green", "red", "pink", "blue", "purple", "black", "orange", "magenta"]
    colors = colors+colors[3:]+colors[3:]+colors[3:]+colors[3:]+colors[3:]+colors[3:]+colors[3:]
    plotter = pv.Plotter()

    Pos_Si = Pos[Types==1]
    def slide(value):
        value = int(round(value))
        L_poly = []
        a=0
        A = []
        for value in range(3,np.max(L_cycles)):
            L_poly = []
            for cycle in Cycles:
                if np.random.random()<0.2:
                    if len(cycle)== value:

                        Pos_cycle = Pos_Si[cycle]
                        Pos_cycle = np.append(Pos_cycle,[Pos_cycle[0]],axis=0)

                        x,y,z = Pos_cycle.transpose()
                        pv_poly = pv.StructuredGrid(x,y,z)
                        L_poly.append(pv_poly)
            if len(L_poly) > 2:
                pv_poly = L_poly[0].merge(L_poly[1:])
                plotter.add_mesh(pv_poly,color=colors[value],line_width=8,name="cycles"+str(value))
                # plotter.add_mesh(pv_poly,color=colors[value],line_width=2,name="cycles"+str(value))



    # print("A")
    a = time.time()
    N_Si = len(Pos_Si)
    Indices = (np.arange(0,N_Si).reshape(N_Si,1,1)*np.array([1,0]) + np.arange(0,N_Si).reshape((1,N_Si,1))*np.array([0,1])).reshape((N_Si*N_Si,2))
    Indices = Indices[Bonds.ravel()!=0]
    print(time.time()-a)


    b = time.time()
    tubes = [pv.Tube(Pos_Si[i_si_1],Pos_Si[i_si_2],n_sides=5,radius=0.2) for i_si_1,i_si_2 in Indices]
    print(time.time()-b)
    b = time.time()
    mesh = tubes[0].merge(tubes[1:])
    print(time.time()-b)
    b = time.time()
    plotter.add_mesh(mesh,opacity=0.05)
    print(time.time()-b)
    L_cycles = [len(cycle) for cycle in Cycles]

    plotter.add_slider_widget(slide, [3,np.max(L_cycles)],value=3,title="Length Cycles", fmt="%3.0f")
    plotter.show()


def find_bridging_oxygens(Pos, Types, threshold_Si=2, do_count_type_3=True):
    """
    For each pair of Si atoms bonded through a bridging O,
    returns a dict: (si_i, si_j) -> local O index (into Pos_O).
    """
    Pos_Si = Pos[Types == 1]
    if do_count_type_3:
        o_mask = (Types == 2) | (Types == 3)
    else:
        o_mask = (Types == 2)
    Pos_O = Pos[o_mask]

    Dist = sd.cdist(Pos_Si, Pos_O)          # (num_Si, num_O)
    Bonds = Dist < threshold_Si             # Si–O bond matrix

    bridging = {}
    for si_i in range(len(Pos_Si)):
        for o_idx in np.where(Bonds[si_i])[0]:
            for si_j in np.where(Bonds[:, o_idx])[0]:
                if si_j != si_i:
                    key = (min(si_i, si_j), max(si_i, si_j))
                    if key not in bridging:          # keep first found
                        bridging[key] = o_idx
    return bridging, Pos_O


def save_cycles(Pos, Types, Cycles, file="cycles.txt",
                threshold_Si=2, do_count_type_3=True):
    """
    Saves cycles in Si–O–Si–O–... order.
    Format per line:
        <n_Si>  si0 si1 ... si_{n-1}  o01 o12 ... o_{n-1,0}
                x0 y0 z0  x_o01 y_o01 z_o01  x1 y1 z1  ...  (interleaved)
    where O indices are global (into the original Pos array).
    """
    L_cycles   = [len(c) for c in Cycles]
    Pos_Si     = Pos[Types == 1]
    o_mask     = (Types == 2) | (Types == 3) if do_count_type_3 else (Types == 2)
    O_global   = np.where(o_mask)[0]          # maps local O idx -> global atom idx

    bridging, Pos_O = find_bridging_oxygens(Pos, Types,
                                             threshold_Si=threshold_Si,
                                             do_count_type_3=do_count_type_3)

    with open(file, 'w') as f:
        f.write("# Cycles file – SHAnC\n")
        f.write("# Format: n_Si  [Si_ids]  [O_global_ids]  "
                "[x y z interleaved: Si0 O01 Si1 O12 ...]\n")

        for j in range(3, np.max(L_cycles) + 1):
            f.write("#cycles of {}\n".format(j))
            for cycle in Cycles:
                if len(cycle) != j:
                    continue

                # collect bridging O (local index) for each edge 
                o_local = []
                for k in range(j):
                    si_i = cycle[k]
                    si_j = cycle[(k + 1) % j]
                    key  = (min(si_i, si_j), max(si_i, si_j))
                    o_local.append(bridging.get(key, -1))   # -1 = not found

                # write: length
                f.write("{} ".format(j))

                # write: Si indices 
                for idx in cycle:
                    f.write("{} ".format(idx))

                # write: O global indices 
                for o_idx in o_local:
                    gid = int(O_global[o_idx]) if o_idx >= 0 else -1
                    f.write("{} ".format(gid))

                # write: interleaved positions Si0 O01 Si1 O12 … 
                for k in range(j):
                    px, py, pz = Pos_Si[cycle[k]]
                    f.write("{:3.3f} {:3.3f} {:3.3f} ".format(px, py, pz))
                    o_idx = o_local[k]
                    if o_idx >= 0:
                        px, py, pz = Pos_O[o_idx]
                    else:
                        px, py, pz = float('nan'), float('nan'), float('nan')
                    f.write("{:3.3f} {:3.3f} {:3.3f} ".format(px, py, pz))
                f.write("\n")
        


def read_cycles(file="cycles.txt"):
    """
    Reads cycles written by the updated save_cycles.

    Returns
    -------
    Cycles     : list of lists of Si indices
    L_cycles   : list of int  (== len of each cycle)
    Pos_cycles : list of (n, 3) arrays with interleaved Si/O positions
                 Row order: Si0, O01, Si1, O12, ..., Si_{n-1}, O_{n-1,0}
    O_ids      : list of lists of global O atom indices
    """
    Cycles, L_cycles, Pos_cycles, O_ids = [], [], [], []

    for line in open(file, "r"):
        if "#" in line:
            continue
        parts  = line.split()
        print(parts)
        n      = int(parts[0])

        si_ids = [int(x) for x in parts[1 : n + 1]]
        o_gids = [int(x) for x in parts[n + 1 : 2 * n + 1]]

        coords = [float(x) for x in parts[2 * n + 1 :]]
        # 2*n atoms (n Si + n O), each with 3 coords → 6*n floats
        pos = np.array(coords).reshape(2 * n, 3)

        Cycles.append(si_ids)
        L_cycles.append(n)
        Pos_cycles.append(pos)
        O_ids.append(o_gids)

    return Cycles, L_cycles, Pos_cycles, O_ids



import os


def get_external_oxygens_validated(Pos, Types, cycle_si_local_indices, bridging_o_indices, threshold=2.0, min_total_o=2):
    """
    Identifies Oxygens attached to the Si atoms of the cycle.
    cycle_si_local_indices : LOCAL indices into the Si-only sub-array (Pos[Types==1])
    """
    # --- FIX: convert local Si indices → global Pos indices ---
    si_global_indices = np.where(Types == 1)[0]
    cycle_si_global = si_global_indices[cycle_si_local_indices]

    o_mask = (Types == 2) | (Types == 3)
    pos_o = Pos[o_mask]
    o_global_indices = np.where(o_mask)[0]

    pos_si_cycle = Pos[cycle_si_global] # now using correct positions

    dist_matrix = sd.cdist(pos_si_cycle, pos_o) # (n_cycle_Si, n_O)

    external_o_gids = []

    for i, si_gidx in enumerate(cycle_si_global):
        local_o_indices = np.where(dist_matrix[i] < threshold)[0]

        if len(local_o_indices) < min_total_o:
            return None # under-coordinated → invalid cycle

        for loc_idx in local_o_indices:
            gid = o_global_indices[loc_idx]
            if gid not in bridging_o_indices:
                external_o_gids.append(gid)

    return list(set(external_o_gids))


# ---------------------------------------------------------------------------
# Cycle extraction and file output helpers
# ---------------------------------------------------------------------------

def extract_cycles_to_files(Cycles, L_cycles, O_ids, Types, Pos,
    target_size=6, threshold=2.0, add_h=False, h_bond_length=0.96, name=None):
    """Extract validated cycles and save them as XYZ fragments."""
    si_global_indices = np.where(Types == 1)[0]
    folder = f"cycle_{target_size}"
    os.makedirs(folder, exist_ok=True)

    count = 1
    for i, size in enumerate(L_cycles):
        # 1. Filter by cycle size.
        if size != target_size:
            continue

        ext_o = get_external_oxygens_validated(
        Pos, Types, Cycles[i], O_ids[i], threshold=threshold
        )
        
        if ext_o is None:
            continue

        # Calculate atom count for filtering
        # atoms = target_size (Si) + len(O_ids) (bridging O) + len(ext_o) (external O)
        n_si = size
        n_o_bridge = len(O_ids[i])
        n_o_ext = len(ext_o)
        total_atoms_no_h = n_si + n_o_bridge + n_o_ext

        # 2. STRICT FILTER: Only keep cycles containing exactly 24 atoms (Si + O)
        if total_atoms_no_h != target_size*4:
            continue

        # Handle optional H passivation on external oxygens.
        h_positions = []
        if add_h:
            h_positions = add_hydrogen_to_oxygens(Pos, Types, ext_o, Cycles[i], bond_length=h_bond_length)

        final_atom_count = total_atoms_no_h + len(h_positions)
        if name is None:
            filename = os.path.join(folder, f"cycle_{size}_{count}.xyz")
        else:
            filename = os.path.join(folder, f"cycle_{name}_{size}_{count}.xyz")
        with open(filename, 'w') as f:
            f.write(f"{final_atom_count}\n")
            f.write(f"Cycle with {total_atoms_no_h} Si+O atoms and {len(h_positions)} H saturators\n")

            # Write cycle Si
            for local_idx in Cycles[i]:
                p = Pos[si_global_indices[local_idx]]
                f.write(f"Si {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")

            # Write bridging O
            for gidx in O_ids[i]:
                p = Pos[gidx]
                f.write(f"O {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")

            # Write external O
            for gidx in ext_o:
                p = Pos[gidx]
                f.write(f"O {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")

            # Write H (if any)
            for hp in h_positions:
                f.write(f"H {hp[0]:.5f} {hp[1]:.5f} {hp[2]:.5f}\n")

        count += 1
    print(f"Extraction complete. Saved {count-1} cycles of total size {target_size} to folder '{folder}'.")


def add_hydrogen_to_oxygens(Pos, Types, external_o_indices, cycle_si_local_indices, bond_length=0.96):
    """
    Adds H to the validated external oxygens. 
    It points the H away from the nearest Si.
    """
    if not external_o_indices:
        return np.array([]).reshape(0, 3)
        
    new_h_pos = []
    si_global_indices = np.where(Types == 1)[0]
    cycle_si_global = si_global_indices[cycle_si_local_indices]
    cycle_si_pos = Pos[cycle_si_global]
    
    for o_idx in external_o_indices:
        o_p = Pos[o_idx]
        # Find the specific Si this O belongs to (directionality)
        dists = np.linalg.norm(cycle_si_pos - o_p, axis=1)
        nearest_si_idx = np.argmin(dists)
        si_p = cycle_si_pos[nearest_si_idx]
        
        # Vector from Si to O
        vec = o_p - si_p
        unit_vec = vec / np.linalg.norm(vec)
        
        # Position H along that vector
        h_p = o_p + (unit_vec * bond_length)
        new_h_pos.append(h_p)
        
    return np.array(new_h_pos)





if __name__=="__main__":

    # file = "quartz_dupl.data"
    # list_BOX,list_ATOMS = read_data(file,do_scale=False)

    # file = "demo/dump_last_oh.lammpstrj"
    # file = "dummp_trimmed.lammpstrj"
    # file = "demo/dummp_128.lammpstrj"
    # file = "demo/dummp_256.lammpstrj"
    # file = "demo/dummp_512.lammpstrj"
    # file = "demo/dummp_1024.lammpstrj"
    # file = "dummps_snad_last.lammpstrj"
    # file = "demo/dummps_round_2_last.lammpstrj"
    file = "sio2/13497439/last_timestep.lammpstrj"
    # file = "sio2/13800979/last_timestep.lammpstrj" # cuboid
    # file = "sio2/Helicetest_P400/last_timestep.lammpstrj"
    # file = "dummps_long_last.lammpstrj"

    # file = "sio2/13800979/cuboid.data"

    # list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file,unscale=True)
    # list_BOX, list_ATOMS = read_data(file)
    # write_dump("cuboid_no_relax.lammpstrj", list_TSTEP=[0],list_NUM_AT=[69774],list_ATOMS=list_ATOMS, list_BOX=np.array(list_BOX))
    # file = "cuboid_no_relax.lammpstrj"
    list_TSTEP, list_NUM_AT, list_BOX, list_ATOMS = read_dump(file,unscale=True)
    list_TSTEP=[0]
    list_Pos = list_ATOMS[:,:,2:]
    list_Types = list_ATOMS[:,:,1]
    Pos = list_ATOMS[-1][:,2:]
    Types = list_ATOMS[-1][:,1]


    if False:
        print("Computing bonds and cycles from scratch...")
        Bonds = compute_bonds_graph(Pos,Types,cube=50,periodic=False,Lims=list_BOX[-1])
        print("Finding cycles...")
        a = time.time()
        Cycles,L_cycles = find_cycles(Bonds)
        Cycles = xor_rm(Cycles)
        save_cycles(Pos, Types, Cycles, file="cycles_cuboid_no_relax.txt")
        print(time.time()-a)
    else:
        Cycles,L_cycles,Pos_cycles, O_ids = read_cycles(file="cycles.txt")
    # Cycles,L_cycles,Pos_cycles, O_ids = read_cycles(file="cycles_cuboid_no_relax.txt")

    from collections import Counter
    count_cycles = Counter(L_cycles)
    total_cycles = len(L_cycles)

    max_type = max(count_cycles, key = count_cycles.get)
    max_count = count_cycles[max_type]
    print(f"{'Cycle Size':<12} | {'Count':<10} | {'Percentage':<12} | {'Relative to Max'}")
    print("-" * 60)
    for size in sorted(count_cycles.keys()):
        count = count_cycles[size]
        percentage = (count / total_cycles) * 100
        relative_to_max = (count / max_count) * 100
        print(f"{size:< 12} | {count:< 10} | {percentage:<11.2f}% | {relative_to_max:>14.2f}%")
    print("-" * 60)
    print(f"Total Cycles: {total_cycles}")
    print(f"Most Common Cycle Size: {max_type} with {max_count} occurrences ({(max_count / total_cycles) * 100:.2f}%)")
    
    EXTRACT_CYCLES = True
    ADD_H = False
    TARGET_SIZE = 6
    BOND_CUTOFF = 1.8

    if EXTRACT_CYCLES:
        if ADD_H:
            extract_cycles_to_files(Cycles, L_cycles, O_ids, Types, Pos, target_size=TARGET_SIZE, add_h=ADD_H, threshold=BOND_CUTOFF)
        else:
            extract_cycles_to_files(Cycles, L_cycles, O_ids, Types, Pos, target_size=TARGET_SIZE, threshold=BOND_CUTOFF)

        # --- Count atoms per cycle of target size ---
        si_global_indices = np.where(Types == 1)[0]
        atom_counts = []

        for i, size in enumerate(L_cycles):
            if size != TARGET_SIZE:
                continue

            ext_o = get_external_oxygens_validated(
                Pos, Types, Cycles[i], O_ids[i], threshold=BOND_CUTOFF
            )
            if ext_o is None:
                continue  # same cycles that were skipped during extraction

            n_atoms = size + len(O_ids[i]) + len(ext_o)
            atom_counts.append(n_atoms)

        atom_counts = np.array(atom_counts)

        print(f"\n--- Cycle size {TARGET_SIZE} atom count statistics ---")
        print(f"Total valid cycles : {len(atom_counts)}")
        print(f"Min atoms          : {atom_counts.min()}")
        print(f"Max atoms          : {atom_counts.max()}")
        print(f"Mean atoms         : {atom_counts.mean():.1f}")
        print(f"Cycles with < 24 atoms: {np.sum(atom_counts < 24)}")

        # Optional: histogram
        unique, counts = np.unique(atom_counts, return_counts=True)
        print("\nAtom count distribution:")
        for u, c in zip(unique, counts):
            print(f"  {u} atoms : {c} cycle(s)")
        



    # L_cycles = [len(cycle) for cycle in Cycles]

    # C = np.zeros((np.max(L_cycles)+3),dtype="int")
    # for k in L_cycles:
    #     C[k]+=1
    # print(C)

    visualize_cycles(Pos,Types,Cycles)
    
    # save_cycles(Pos,Types,Cycles,file="cycles_60ps.txt")
    # plot_cycles(L_cycles)
    # G = nx.from_numpy_array(Bonds)
    # print(len(Cycles),(G.number_of_edges() - G.number_of_nodes() + 1))

  