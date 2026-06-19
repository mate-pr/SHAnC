import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Construction.import_libraries import *
from Construction.distortion import *
from Construction.analysis import *
from Construction.read_write import *
from Construction.cycle_extraction import *


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


print("Computing bonds and cycles from scratch...")
Bonds = compute_bonds_graph(Pos,Types,cube=50,periodic=False,Lims=list_BOX[-1])
print("Finding cycles...")
a = time.time()
Cycles,L_cycles = find_cycles(Bonds)
Cycles = xor_rm(Cycles)
save_cycles(Pos, Types, Cycles, file="cycles.txt")
print(time.time()-a)

Cycles,L_cycles,Pos_cycles, O_ids = read_cycles(file="cycles.txt")

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

EXTRACT_CYCLES = False
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

visualize_cycles(Pos,Types,Cycles, list_BOX)

