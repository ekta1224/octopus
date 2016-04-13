#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pygadgetreader import *
import sys

V_radius = 2.0 # Radius of disk particles to compute the CM

#Function that computed the MW CM using the disk potential

def CM_disk_potential(x, y, z, vx, vy, vz, Pdisk):
    min_pot = np.where(Pdisk==min(Pdisk))[0]
    x_min = x[min_pot]
    y_min = y[min_pot]
    z_min = z[min_pot]
    # This >2.0 corresponds to the radius in kpc of the particles that
    # I am taking into account to compute the CM
    avg_particles = np.where(np.sqrt((x-x_min)**2.0 + (y-y_min)**2.0 +
(z-z_min)**2.0)<V_radius)[0]
    x_cm = sum(x[avg_particles])/len(avg_particles)
    y_cm = sum(y[avg_particles])/len(avg_particles)
    z_cm = sum(z[avg_particles])/len(avg_particles)
    vx_cm = sum(vx[avg_particles])/len(avg_particles)
    vy_cm = sum(vy[avg_particles])/len(avg_particles)
    vz_cm = sum(vz[avg_particles])/len(avg_particles)
    return x_cm, y_cm, z_cm, vx_cm, vy_cm, vz_cm

# Function that computes the CM iterativly
def CM(xyz, vxyz, delta=0.0333):
    """
    Compute the center of mass coordinates and velocities of a halo.
    It iterates in radii until reach a convergence given by delta.

    Parameters:
    -----------
    xyz: cartesian coordinates with shape (n,3)
    vxys: cartesian velocities with shape (n,3)
    delta(optional): Precision of the CM, D=0.033

    Returns:
    --------
    xcm, ycm, zcm, vxcm, vycm, vzcm: coordinates and velocities of
    the center of mass with reference to a (0,0,0) point.

    """
    N = len(xyz)
    xCM = sum(xyz[:,0])/N
    yCM = sum(xyz[:,1])/N
    zCM = sum(xyz[:,2])/N

    xCM_new = xCM
    yCM_new = yCM
    zCM_new = zCM

    xCM = 0.0
    yCM = 0.0
    zCM = 0.0

    vxCM_new = sum(vxyz[:,0])/N
    vyCM_new = sum(vxyz[:,1])/N
    vzCM_new = sum(vxyz[:,2])/N

    R1 = np.sqrt((xyz[:,0] - xCM_new)**2 + (xyz[:,1] - yCM_new)**2 + (xyz[:,2] - zCM_new)**2)
    i=0
    while (np.sqrt((xCM_new-xCM)**2 + (yCM_new-yCM)**2 +(zCM_new-zCM)**2) > delta):
        xCM = xCM_new
        yCM = yCM_new
        zCM = zCM_new
        Rcm = np.sqrt(xCM**2 + yCM**2 + zCM**2)
        R = np.sqrt((xyz[:,0] - xCM)**2 + (xyz[:,1] - yCM)**2 + (xyz[:,2] - zCM)**2)
        Rmax = np.max(R)
        index = np.where(R<Rmax/1.3)[0]
        xyz = xyz[index]
        vxyz = vxyz[index]
        N = len(xyz)
        i+=1
        xCM_new = np.sum(xyz[:,0])/N
        yCM_new = np.sum(xyz[:,1])/N
        zCM_new = np.sum(xyz[:,2])/N
        vxCM_new = np.sum(vxyz[:,0])/N
        vyCM_new = np.sum(vxyz[:,1])/N
        vzCM_new = np.sum(vxyz[:,2])/N
        #Rnow[i] = max(np.sqrt((x - xCM_new[i])**2 + (y - yCM_new[i])**2 + (z - zCM_new[i])**2))
    #clean = np.where(Rnow != 0)[0]
    return xCM_new, yCM_new, zCM_new, vxCM_new, vyCM_new, vzCM_new

# function that computes the CM using the 10% most bound particles!
# I am using the potential method any more, its not useful to find the 
#LMC CM because the particles feel the potential of the MW.
"""
def potential_CM(potential, x, y, z, vx, vy, vz):
    index = np.where(potential< min(potential)*0.90)[0]
    x_p = x[index]
    y_p = y[index]
    z_p = z[index]
    vx_p = vx[index]
    vy_p = vy[index]
    vz_p = vz[index]
    N = len(x_p)
    x_cm = sum(x_p)/N
    y_cm = sum(y_p)/N
    z_cm = sum(z_p)/N
    vx_cm = sum(vx_p)/N
    vy_cm = sum(vy_p)/N
    vz_cm = sum(vz_p)/N
    Rcm = np.sqrt(x_cm**2.0 + y_cm**2.0 + z_cm**2.0)
    Vcm = np.sqrt(vx_cm**2.0 + vy_cm**2.0 + vz_cm**2.0)
    return x_cm, y_cm, z_cm, vx_cm, vy_cm, vz_cm, Rcm, Vcm


#Function that computes the CM of the halo using the minimum of the
#potential:
def CMMW(x, y, z, pot):
    rcut = np.where(np.sqrt(x**2+y**2+z**2)<30.0)[0]
    x, y, z, pot = x[rcut], y[rcut], z[rcut], pot[rcut]
    cm = np.where(pot == min(pot))[0]
    x_cm, y_cm, z_cm = x[cm], y[cm], z[cm]
    return x_cm, y_cm, z_cm

def CMLMC(x, y, z, pot, xcmmw, ycmmw, zcmmw):
    xcm = sum(x)/len(x)
    ycm = sum(y)/len(y)
    zcm = sum(z)/len(z)
    rcut = np.where(np.sqrt((x-xcm)**2+(y-ycm)**2+(z-zcm)**2)<20.0)[0]
    x, y, z, pot = x[rcut], y[rcut], z[rcut], pot[rcut]
    cm = np.where(pot == min(pot))[0]
    x_cm, y_cm, z_cm = x[cm], y[cm], z[cm]
    return x_cm, y_cm, z_cm

def VCM(x, y, z, xcm, ycm, zcm, vx, vy, vz):
    Ntot = len(x)
    N = Ntot
    while(N>0.1*Ntot):
        rshell = np.sqrt((x-xcm)**2 + (y-ycm)**2 + (z-zcm)**2)
        rcut = max(rshell) / 1.1
        cut = np.where(rshell<=rcut)[0]
        x, y, z = x[cut], y[cut], z[cut]
        vx, vy, vz = vx[cut], vy[cut], vz[cut]
        N = len(x)
    vxcm = sum(vx)/N
    vycm = sum(vy)/N
    vzcm = sum(vz)/N
    return vxcm, vycm, vzcm
"""

