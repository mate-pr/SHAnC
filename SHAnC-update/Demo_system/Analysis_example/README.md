# Analysis Example

This folder contains example scripts and data for analyzing silica nanohelix surface properties and cycle statistics.

## Contents

- `Analyze_surface.py` - example analysis workflow for `sio2_pitch_200` systems.
- `Extract_cycles.py` - utility scripts to extract ring cycles from topology and bond graphs.
- `Structure_validation.py` - validation helper scripts for structure quality and bonding.
- `au_pitch_200/` - example gold-coated helix data.
- `sio2_pitch_200/` - example silica helix data.

## Purpose

The example notebook/script in this folder demonstrates how to:

- load a LAMMPS dump trajectory with `read_dump`
- identify surface atoms using `external_surface`
- compute surface area with multiple methods
- visualize contour slices and cast/surface atoms
- assess undercoordinated Si/O sites and surface chemistry

## Usage

1. Open `Analyze_surface.py`.
2. Adjust the target system folder and parameter preset if needed.
3. Run the script in the `System_test/Analysis_example` root.

## Surface Area Methods

The example compares several surface area calculations:

- Method 1: atom-count based estimate
- Method 2: slice perimeter integration
- Method 3: circle perimeter difference
- Method 4: tubular surface approximation
- Method 5: transformed cuboid mesh estimate

## Notes

- The script uses the `Construction` package utilities from the parent directory.
- Comments and documentation style in this folder are aligned to the `Construction` package: descriptive docstrings, clear parameter lists, and method purpose statements.
