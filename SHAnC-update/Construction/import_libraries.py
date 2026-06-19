"""
Importation of the requiered libraries
"""

import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from matplotlib.widgets import Slider
import scipy as sp
import scipy.spatial.distance as sd
import scipy.signal as sps
import matplotlib.ticker as ticker
import re
from collections import defaultdict
import os
import time
import networkx as nx

# Core numerical and geometry utilities
# - numpy: array maths and reshaping
# - scipy.spatial.KDTree: fast neighbour searches for atom pairs
# - scipy.spatial.distance: distance matrices and pairwise distances
# - scipy.signal: signal and histogram processing
# - matplotlib: plotting and interactive widgets
# - pyvista: 3D structure rendering and isosurfaces
# - re: parsing text in XYZ/LAMMPS file headers
# - defaultdict: convenient dictionary defaults for counters and maps

