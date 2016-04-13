"""
Code to compute the density from a distribution of dots.
This implement the SPH Kernel Method (Hernquist & Katz 89)

"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "nico-garavito <jngaravitoc@email.arizona.edu>"

import numpy as np
import sys
from sklearn.neighbors import KDTree
import warnings
warnings.filterwarnings('ignore')


# Finding the nearest 32 neihboors
def nearest_neighboors(x, y, z, r):
    D = np.array([x, y, z])
    D = D.T
    tree = KDTree(D, leaf_size=2500)
    dist, ind = tree.query(r, k=33)
    return dist[0], ind[0]

#Evaluate the Kernel
def kernel(r, h):
    if r<h:
        W = 1. - 3./2.*(r/h)**2.0 + 3./4.*(r/h)**3.0
    elif ((r>h) & (r<2.0*h)):
        W = 1./4. * (2. - r/h)**3.
    else:
        W = 0.0
    return W/(np.pi*h**3.0)

#Compute the local density
def density(x,y,z, mass, r):
    dn, idn = nearest_neighboors(x, y, z, r)
    h = np.max(dn)/2.0
    rho = np.zeros(33.0)
    m = mass[idn]
    for i in range(len(dn)):
        W = kernel(dn[i], h)
        rho[i] = m[i]*W
    return np.sum(rho)

#Making a grid with densities
def grid(X, Y, Z, res):
    """
    Make a grid with all the densities.
    :param X, Y, Z:
         Coordinates of the dots.
    :param res:
         Resolution of the square grid.
    :return rho:
         A 2d array with the densities.
    """
    mass = np.ones(len(X))
    rho = np.zeros((res, res))
    rx = np.linspace(min(X)+min(X)*0.2, max(X)+max(X)*0.2, res)
    ry = np.linspace(min(Y)+min(Y)*0.2, max(Y)+max(Y)*0.2, res)
    for i in range(res):
        for j in range(res):
            rho[i][j] = density(X, Y, Z, mass, [rx[i], ry[j], 0])
    return rho


