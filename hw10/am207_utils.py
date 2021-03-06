"""
Harvard AM 207

Michael S. Emanuel
Fri Sep 21 17:16:18 2018
"""

import numpy as np
import matplotlib as mpl
import pickle
from typing import Dict, Callable

# *************************************************************************************************
# Type aliases
funcType = Callable[[float], float]

# *************************************************************************************************
def plot_style() -> None:
    """Set plot style for the session."""
    # Set default font size to 20
    mpl.rcParams.update({'font.size': 20})
    
    
# *************************************************************************************************
def range_inc(x: int, y: int = None, z: int = None) -> range:
    """Return a range inclusive of the end point, i.e. range(start, stop + 1, step)"""
    if y is None:
        (start, stop, step) = (1, x + 1, 1)
    elif z is None:
        (start, stop, step) = (x, y + 1, 1)
    elif z > 0:
        (start, stop, step) = (x, y + 1, z)
    elif z < 0:
        (start, stop, step) = (x, y - 1, z)
    return range(start, stop, step)


def arange_inc(x: float, y: float = None, z: float = None) -> np.ndarray:
    """Return a numpy arange inclusive of the end point, i.e. range(start, stop + 1, step)"""
    if y is None:
        (start, stop, step) = (1, x + 1, 1)
    elif z is None:
        (start, stop, step) = (x, y + 1, 1)
    elif z > 0:
        (start, stop, step) = (x, y + 1, z)
    elif z < 0:
        (start, stop, step) = (x, y - 1, z)
    return np.arange(start, stop, step)


# *************************************************************************************************
def load_vartbl(fname: str) -> Dict:
    """Load a dictionary of variables from a pickled file"""
    try:
        with open(fname, 'rb') as fh:
            vartbl = pickle.load(fh)
    except:
        vartbl = dict()
    return vartbl


def save_vartbl(vartbl: Dict, fname: str) -> None:
    """Save a dictionary of variables to the given file with pickle"""
    with open(fname, 'wb') as fh:
        pickle.dump(vartbl, fh)
