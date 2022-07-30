'''
Functions for performing the spherical
harmonic expansion of the potential
'''
import numpy as np
import math as mt
from scipy.special import lpmv
from scipy import linalg, matrix
import scipy
# import pyshtools

def legendre(n, X):
    '''
    Returns a list of all the associated legendre functions of degree n
    and order m = 0, 1.... n for each element in X
    '''
    r = []
    for m in range(n + 1):
        r.append(lpmv(m, n, X))
    return r

def spher_harm_basis(r0, X, Y, Z, order):
    '''
    Computes spherical harmonics.
   
    Returns: 
    - Yxx, a 1D array of the spherical harmonic evaluated on the grid.
    - rnorm, a normalization factor for the spherical harmonics.

    The function returns the coefficients in the order: [C00 C10 C11c C11s]
    These correspond to the multipoles in cartesian coordinates: 
    [c z -x -y (z^2 - x^2 / 2 - y^2 / 2)  -2 * 3zx  -2 * 3yz  2 * (3x^2 - 3y^2)  2 * 6xy .
     1 2  3  4           5                   6         7              8             9  ..
     The coefficients for higher order multipoles should be checked!!


    Or in terms of the Littich thesis:
    M1 M3 M4 M2 M7 M8 M6 M9 M5 (Using the convention in G. Littich's master thesis (2011))
    0  1  2  3  4  5  6  7  8  (the ith component of Q matrix)
    '''

    # initialize naught coordinate values. grid with expansion point (r0) at 0.
    x0, y0, z0 = r0
    
    # Number of points.
    nx = len(X)
    ny = len(Y)
    nz = len(Z)
    npts = nx * ny * nz

    # Create 3D arrays with number of points defined by distance to center. Why is y and x flipped?
    y, x, z = np.meshgrid(Y - y0, X - x0, Z - z0)
    # Flatten these 3D arrays.
    x, y, z = np.reshape(x, npts), np.reshape(y, npts), np.reshape(z, npts)

    # Change variables to spherical coordinates.
    r = np.sqrt(x * x + y * y + z * z)
    r_trans = np.sqrt(x * x + y * y)
    theta = np.arctan2(r_trans, z)
    phi = np.arctan2(y, x)

    ## This code is not used.
    # For now normalizing as in matlab code
    dl = Z[1] - Z[0]

    
    # r scaled with normalization factor.
    scale = 1 ## Could be this: np.sqrt(np.amax(r) * dl)
    rs = r / (scale)

    # The expansion coefficients to return.
    Q = []
    Q.append(np.ones(npts)) # Does using .ones have to change if the normalization factor is different?

    # Real part of spherical harmonics.
    for n in range(1, order + 1):
        # List of all the cos legendre polynomials up to n.
        p = legendre(n, np.cos(theta))

        # Constant term.
        c = (rs ** n) * p[0]
        Q.append(c)

        # Iteration creates the 9th order multipole (i + 1) ** 2. ? Not entirely sure what this code does.
        for m in range(1, n + 1):
            c = (rs ** n) * p[m] * np.cos(m * phi)
            Q.append(c)
            cn = (rs ** n) * p[m] * np.sin(m * phi)
            Q.append(cn)

    # Ill - defined?
    # Converts Q from a list of arrays (for each point) to an array (rows=number of points, col=number of multipoles).
    # This array contains the multipole coefficients up to the 9th order for each point.
    Q = np.transpose(Q)

    return Q, scale

def spher_harm_expansion(potential_grid, r0, X, Y, Z, order):
    '''
    Compute the least-squares solution for the spherical harmonic expansion on potential_grid.
    Arguments:
    potential_grid: 3D array of potential values
    r0: list [x0, y0, z0] of the expansion point
    X, Y, Z: axis ranges for the potential grid
    order: int, order of the expansion
    '''
    # Convert the 3D DC potential into 1D array.
    # Numerically invert, here the actual expansion takes place and we obtain the expansion coefficients M_{j}.

    # Number of of coordinates.
    nx = len(X)
    ny = len(Y)
    nz = len(Z)
    
    # Total number of points.
    npts = nx * ny * nz

    # Flatten the grid of potential values from the RF electrode.
    W = np.reshape(potential_grid, npts)
    # Transpose to column array
    W = np.array([W]).T 

    Yj, scale = spher_harm_basis(r0, X, Y, Z, order)
    # Yj, rnorm = spher_harm_basis_v2(r0, X, Y, Z, order)

    # This is solving the equation 2.95 for the M matrix in the thesis.
    Mj = np.linalg.lstsq(Yj, W, rcond = None)
    # The lstsq function returns 4 items, so we can select the actual solution, which
    # is the first item.
    Mj = Mj[0]

    # Rescale to original units
    i = 0
    for n in range(1, order + 1):
        for m in range(1, 2 * n + 2):
            i  += 1
            Mj[i] = Mj[i] / (scale ** n)
    return Mj, Yj, scale
 
def spher_harm_cmp(Mj, Yj, scale, order):
    '''
    Regenerates the potential (V) from the spherical harmonic coefficients. 
    '''
    V = []
    # unnormalize
    i = 0
    for n in range(1, order + 1):
        for m in range(1, 2 * n + 2):
            i  += 1
            Mj[i] = Mj[i] * (scale ** n)
    
    # In this case, Mj is a col vector. Yj is a 2D array with rows being points and col being coefficients.
    W = np.dot(Yj, Mj)
    return np.real(W)

def nullspace(A, eps = 1e-15):
    u, s, vh = np.linalg.svd(A)
    nnz = (s >= eps).sum()
    null_space = vh[nnz:].conj().T
    return null_space




