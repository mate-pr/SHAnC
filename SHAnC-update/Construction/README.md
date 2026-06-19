# Construction

The `Construction` package contains the core utilities for building and analyzing nanohelix systems.

## Modules

- `import_libraries.py` — central import file for the scientific dependencies used by the package.
- `distortion.py` — lattice duplication, helix generation, and transformation utilities.
- `read_write.py` — functions for reading and writing structure files such as XYZ and LAMMPS data.
- `analysis.py` — structural analysis routines, bond counting, validation, and radial distribution functions.
- `add_chromophore.py` — surface extraction and chromophore grafting utilities.
- `cycle_extraction.py` — ring and cycle extraction helpers with related visualization support.

## Usage

Import the package from the repository root and call the provided functions. Example:

```python
from Construction.distortion import create_syst
```

## Purpose

This package provides the reusable building blocks for creating silica nanohelix geometries, analyzing their structure, and attaching functional molecules to reactive surface sites.
