#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pygadgetreader import *


#Function that computed the MW CM using the disk potential

def MW_LMC_particles(xyz, vxyz, pids, NMW_particles):
    """
    Function that return the MW and the LMC particles
    positions and velocities.

    Parameters:
    -----------
    xyz: snapshot coordinates with shape (n,3)
    vxys: snapshot velocities with shape (n,3)
    pids: particles ids
    NMW_particles: Number of MW particles in the snapshot
    Returns:
    --------
    xyz_mw, vxyz_mw, xyzlmc, vxyz_lmc: coordinates and velocities of
    the MW and the LMC.

    """
    sort_indexes = np.sort(pids)
    N_cut = sort_indexes[NMW_particles]
    MW_ids = np.where(pids<N_cut)[0]
    LMC_ids = np.where(pids>=N_cut)[0]
    return xyz[MW_ids], vxyz[MW_ids], xyz[LMC_ids], vxyz[LMC_ids]

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

def CM(xyz, vxyz, delta=0.025):
    """
    Compute the center of mass coordinates and velocities of a halo
    using the Shrinking Sphere Method Power et al 2003.
    It iterates in radii until reach a convergence given by delta
    or 1% of the total number of particles.

    Parameters:
    -----------
    xyz: cartesian coordinates with shape (n,3)
    vxys: cartesian velocities with shape (n,3)
    delta(optional): Precision of the CM, D=0.025

    Returns:
    --------
    rcm, vcm: 2 arrays containing the coordinates and velocities of
    the center of mass with reference to a (0,0,0) point.

    """
    N_i = len(xyz)
    N = N_i

    xCM = 0.0
    yCM = 0.0
    zCM = 0.0

    xCM_new = sum(xyz[:,0])/N_i
    yCM_new = sum(xyz[:,1])/N_i
    zCM_new = sum(xyz[:,2])/N_i

    vxCM_new = sum(vxyz[:,0])/N_i
    vyCM_new = sum(vxyz[:,1])/N_i
    vzCM_new = sum(vxyz[:,2])/N_i

    while ((np.sqrt((xCM_new-xCM)**2 + (yCM_new-yCM)**2 + (zCM_new-zCM)**2) > delta) & ((N>N_i*0.01) | (N>1000))):
        xCM = xCM_new
        yCM = yCM_new
        zCM = zCM_new
        # Re-centering sphere
        xyz[:,0] = xyz[:,0]
        xyz[:,1] = xyz[:,1]
        xyz[:,2] = xyz[:,2]
        R = np.sqrt((xyz[:,0]-xCM_new)**2 + (xyz[:,1]-yCM_new)**2 + (xyz[:,2]-zCM_new)**2)
        Rmax = np.max(R)
        # Reducing Sphere by its 2.5%
        index = np.where(R<Rmax*0.975)[0]
        xyz = xyz[index]
        vxyz = vxyz[index]
        N = len(xyz)
        #Computing new CM coordinates and velocities
        xCM_new = np.sum(xyz[:,0])/N
        yCM_new = np.sum(xyz[:,1])/N
        zCM_new = np.sum(xyz[:,2])/N
        vxCM_new = np.sum(vxyz[:,0])/N
        vyCM_new = np.sum(vxyz[:,1])/N
        vzCM_new = np.sum(vxyz[:,2])/N
    return np.array([xCM_new, yCM_new, zCM_new]), np.array([vxCM_new, vyCM_new, vzCM_new])



def orbit(path, snap_name, initial_snap, final_snap, NMW_particles, delta, lmc=False):
    """
    Computes the orbit of the MW and the LMC. It compute the CM of the
    MW and the LMC using the shrinking sphere method at each snapshot.

    Parameters:
    -----------
    path: Path to the simulation snapshots
    snap_name: Base name of the snaphot without the number and
    file type, e.g: LMCMW
    initial_snap: Number of the initial snapshot
    final_snap: Number of the final snapshot
    NMW_particles: Number of MW particles in the simulation.
    delta: convergence distance
    lmc: track the lmc orbit. (default = False)
    Returns:
    --------
    XMWcm, vMWcm, xLMCcm, vLMCcm: 4 arrays containing the coordinates
    and velocities of the center of mass with reference to a (0,0,0) point
    at a given time.

    """

    N_snaps = final_snap - initial_snap + 1
    MW_rcm = np.zeros((N_snaps,3))
    MW_vcm = np.zeros((N_snaps,3))
    LMC_rcm = np.zeros((N_snaps,3))
    LMC_vcm = np.zeros((N_snaps,3))

    for i in range(initial_snap, final_snap+1):
        xyz = readsnap(path + snap_name + '_{:03d}.hdf5'.format(i),'pos', 'dm')
        vxyz = readsnap(path + snap_name +'_{:03d}.hdf5'.format(i),'vel', 'dm')
        pids = readsnap(path + snap_name +'_{:03d}.hdf5'.format(i),'pid', 'dm')
        if lmc==True:
            MW_xyz, MW_vxyz, LMC_xyz, LMC_vxyz = MW_LMC_particles(xyz, vxyz, pids, NMW_particles)
            MW_rcm[i], MW_vcm[i] = CM(MW_xyz, MW_vxyz, delta)
            LMC_rcm[i], LMC_vcm[i] = CM(LMC_xyz, LMC_vxyz, delta)
        else:
            MW_rcm[i], MW_vcm[i] = CM(xyz, vxyz, delta)
    return MW_rcm, MW_vcm, LMC_rcm, LMC_vcm
