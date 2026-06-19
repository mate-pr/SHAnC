# SHAnC_update

Update of the previous SHAnC code for building, analyzing, and (added) grafting molecules onto silica nanohelix systems.

## Repository layout

- `Construction/` — core Python modules for geometry generation, file I/O, analysis, and surface attachment.
- `System_test/` — example scripts, test harnesses, and lattice data for building and validating systems.
- `System_test/files_lattices/` — shared lattice files, including metal and silica lattice data.
- `System_test/files_lattices/Model_file/` — the original `beta_quartz.data` used as the SiO2 input for nanohelix construction.

## Quick start

1. From the repository root:

```bash
cd /Users/mtemey/Documents/stage2026MT/code/SHAnC_clean
python3 System_test/Create_a_nanohelix.py
```

2. The script imports the local `Construction` package and writes lattice output data.

## Dependencies

The project uses scientific Python packages such as:

- `numpy`
- `scipy`
- `matplotlib`

Additional optional modules may include visualization or analysis libraries depending on the code path.

## Notes

This repository is structured for rapid development and testing of silica helix construction and molecular grafting workflows.

