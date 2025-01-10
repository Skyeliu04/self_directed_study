import numpy as np

def read_matrix(filename):
    """Read a matrix from a text file."""
    return np.loadtxt(filename) 