# Demo_system

This directory contains example test scripts and lattice data used to verify and demonstrate nanohelix construction.

## Key files

- `Create_a_nanohelix.py` — test runner that builds a silica nanohelix from an input lattice and writes output data files.
- `cuboid.data` — example lattice data file for cuboid geometry.
- `in_template.lmp` — LAMMPS template file used for data conversion or simulation setup.
- `quartz_dupl.data` — output from the nanohelix creation workflow.
- `quartz_int.data` — output interior lattice file from the same workflow.

## Running the test

Execute the example from this directory or from the repository root:

```bash
cd /Users/mtemey/Documents/stage2026MT/code/SHAnC_clean/System_test
python3 Create_a_nanohelix.py
```

## Notes

The example script imports the local `Construction` package and reads the input silica lattice from `files_lattices/Model_file/beta_quartz.data`.

