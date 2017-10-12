#!/sr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from pygadgetreader import *
import matplotlib
matplotlib.use('Agg')
#mport matplotlib.pyplot as plt 

import matplotlib.pyplot as plt

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

def CM_disk_potential(xyz, vxyz, Pdisk): 
    V_radius = 2
    vx = vxyz[:,0]
    vy = vxyz[:,1]
    vz = vxyz[:,2]
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    min_pot = np.where(Pdisk==min(Pdisk))[0]
    x_min = x[min_pot]
    y_min = y[min_pot]
    z_min = z[min_pot]
    # This >2.0 corresponds to the radius in kpc of the particles that
    # I am taking into account to compute the CM
    avg_particles = np.where(np.sqrt((x-x_min)**2.0 + (y-y_min)**2.0 + (z-z_min)**2.0)<V_radius)[0]
    x_cm = sum(x[avg_particles])/len(avg_particles)
    y_cm = sum(y[avg_particles])/len(avg_particles)
    z_cm = sum(z[avg_particles])/len(avg_particles)
    vx_cm = sum(vx[avg_particles])/len(avg_particles)
    vy_cm = sum(vy[avg_particles])/len(avg_particles)
    vz_cm = sum(vz[avg_particles])/len(avg_particles)
    return np.array([x_cm, y_cm, z_cm]), np.array([vx_cm, vy_cm, vz_cm])

def velocities_r15(cm_pos, pos, vel):
    """
    Function to compute the COM velocity in a sphere of 15 kpc
    """
    # Compute the distance with respect to the COM
    R_cm = ((pos[:,0]-cm_pos[0])**2 + (pos[:,1]-cm_pos[1])**2 + (pos[:,2]-cm_pos[2])**2)**0.5
    # Select the particles inside 15 kpc
    index = np.where(R_cm < 15)[0]
    # Compute the velocities of the COM:
    velx_cm = sum(vel[index,0])/len(vel[index,0])
    vely_cm = sum(vel[index,1])/len(vel[index,1])
    velz_cm = sum(vel[index,2])/len(vel[index,2])

    return velx_cm, vely_cm, velz_cm




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


    while (((np.sqrt((xCM_new-xCM)**2 + (yCM_new-yCM)**2 + (zCM_new-zCM)**2) > delta) & (N>N_i*0.01)) | (N>1000)):
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
        N = len(xyz)
        #Computing new CM coordinates and velocities
        xCM_new = np.sum(xyz[:,0])/N
        yCM_new = np.sum(xyz[:,1])/N
        zCM_new = np.sum(xyz[:,2])/N

    vxCM_new, vyCM_new, vzCM_new = velocities_r15([xCM_new, yCM_new, zCM_new], xyz, vxyz)
    return np.array([xCM_new, yCM_new, zCM_new]), np.array([vxCM_new, vyCM_new, vzCM_new])


def ss_velocities(cm_pos, pos, vel, delta):
    N_i = len(vel)
    N = N_i

    vxCM = 0
    vyCM = 0
    vzCM = 0
    vxCM_new = np.zeros(1000)
    vyCM_new = np.zeros(1000)
    vzCM_new = np.zeros(1000)
    vxCM_new[1] = sum(vel[:,0])/N_i
    vyCM_new[1] = sum(vel[:,1])/N_i
    vzCM_new[1] = sum(vel[:,2])/N_i
    i = 1
    Rmax = []
    Rmax.append(100)
    R = np.sqrt((pos[:,0]-cm_pos[0])**2 + (pos[:,1]-cm_pos[1])**2 + (pos[:,2]-cm_pos[2])**2)

    #while ((np.sqrt((vxCM_new[i]-vxCM_new[i-1])**2 + (vyCM_new[i]-vyCM_new[i-1])**2 + (vzCM_new[i]-vzCM_new[i-1])**2) > delta) | (N>1000)):
    while (N>1000):
        # Reducing Sphere by its 2.5%
        index = np.where(R<Rmax[i-1]*0.9)[0]
        R = R[index]
        Rmax.append(np.max(R))
        #print(len(index))
        vel = vel[index]
        #print(len(vel))
        N = len(vel)
        #print(np.max(R), N)
        i+=1

        #Computing new CM coordinates and velocities
        vxCM_new[i] = np.sum(vel[:,0])/N
        vyCM_new[i] = np.sum(vel[:,1])/N
        vzCM_new[i] = np.sum(vel[:,2])/N

        #print(i)
        #print(Rmax[i])

    return  vxCM_new[np.nonzero(vxCM_new)], vyCM_new[np.nonzero(vyCM_new)], vzCM_new[np.nonzero(vzCM_new)], Rmax


def plot_velocities(vxcm, vycm, vzcm, R, i):
    plt.figure(figsize=(14,6))
    plt.subplot(1, 3, 1)
    plt.plot(R, vxcm, label='$vx_{cm}$')
    plt.legend()
    plt.ylabel('$v$[Km/s]')
    plt.xlabel('$R_{max}$[Kpc]')

    plt.subplot(1, 3, 2)
    plt.plot(R, vycm, label='$vy_{cm}$')
    plt.legend()
    plt.xlabel('$R_{max}$[Kpc]')

    plt.subplot(1, 3, 3)
    plt.plot(R, vzcm, label='$vz_{cm}$')
    plt.xlabel('$R_{max}$[Kpc]')
    plt.legend()

    plt.xlabel('$R_{max}$[Kpc]')
    plt.savefig('velocities_LMC3_COM_snap_{:.0f}.png'.format(i), bbox_inches='tight', dpi=300)

def orbit(path, snap_name, initial_snap, final_snap, NMW_particles, delta, lmc=False, disk=False):
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
        # Loading the data!
        xyz = readsnap(path + snap_name + '_{:03d}.hdf5'.format(i),'pos', 'dm')
        vxyz = readsnap(path + snap_name +'_{:03d}.hdf5'.format(i),'vel', 'dm')
        pids = readsnap(path + snap_name +'_{:03d}.hdf5'.format(i),'pid', 'dm')

        if (disk==True):
            MW_xyz_disk = readsnap(path + snap_name + '_{:03d}.hdf5'.format(i),'pos', 'disk')
            MW_vxyz_disk = readsnap(path + snap_name + '_{:03d}.hdf5'.format(i),'vel', 'disk')
            MW_pot_disk = readsnap(path + snap_name + '_{:03d}.hdf5'.format(i),'pot', 'disk')

        ## computing COM
        if lmc==True:
            MW_xyz, MW_vxyz, LMC_xyz, LMC_vxyz = MW_LMC_particles(xyz, vxyz, pids, NMW_particles)
            if disk==True:
                MW_rcm[i-initial_snap], MW_vcm[i-initial_snap] = CM_disk_potential(MW_xyz_disk, MW_vxyz_disk, MW_pot_disk)
            else:
                MW_rcm[i-initial_snap], MW_vcm[i-initial_snap] = CM(MW_xyz, MW_vxyz, delta)
            LMC_rcm[i-initial_snap], LMC_vcm[i-initial_snap] = CM(LMC_xyz, LMC_vxyz, delta)
            LMC_vx, LMC_vy, LMC_vz, R_shell = ss_velocities(LMC_rcm[i-initial_snap], LMC_xyz, LMC_vxyz, 0.5)
            #plot_velocities(LMC_vx, LMC_vy, LMC_vz, R_shell, i-initial_snap)

        else:
            if disk==True:
                MW_rcm[i-initial_snap], MW_vcm[i-initial_snap] = CM_disk_potential(MW_xyz_disk, MW_vxyz_disk, MW_pot_disk)
            else:
                MW_rcm[i-initial_snap], MW_vcm[i-initial_snap] = CM(xyz, vxyz, delta)

    return MW_rcm, MW_vcm, LMC_rcm, LMC_vcm


def write_orbit(filename, MWpos, MWvel, LMCpos, LMCvel):
    f = open(filename, 'w')

    f.write('# MW x(kpc), MW y(kpc), MW z(kpc), MW vx(km/s), MW vy(km/s), MW vz(km/s) LMC x(kpc), LMC y(kpc), LMC z(kpc), LMC vx(km/s), LMC vy(km/s), LMC vz(km/s) \n')

    for i in range(len(MWpos[:,0])):
        f.write(("{:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} \n").format(MWpos[i,0],\
                 MWpos[i,1], MWpos[i,2], MWvel[i,0], MWvel[i,1], \
                 MWvel[i,2], LMCpos[i,0], LMCpos[i,1], \
                 LMCpos[i,2], LMCvel[i,0], LMCvel[i,1], LMCvel[i,2]))

    f.close()

if __name__ == "__main__":

   if (len(sys.argv)<10):
       print('usage: path snap_name initial_snap final_snap NMW_particlesm delta lmc disk')
       exit(0)

   path = sys.argv[1]
   snap_name = sys.argv[2]
   initial_snap = int(sys.argv[3])
   final_snap = int(sys.argv[4])
   NMW_particles = int(sys.argv[5])
   delta = float(sys.argv[6])
   lmc = int(sys.argv[7])
   disk = int(sys.argv[8])
   out_name = sys.argv[9]

   if ((lmc==1) & (disk==1)):
       MW_rcm, MW_vcm, LMC_rcm, LMC_vcm = orbit(path, snap_name, initial_snap, final_snap,
                                                 NMW_particles, delta, lmc=True, disk=True)

   if ((lmc==1) & (disk==0)):
       MW_rcm, MW_vcm, LMC_rcm, LMC_vcm = orbit(path, snap_name, initial_snap, final_snap,
                                                 NMW_particles, delta, lmc=True, disk=False)

   if ((lmc==0)& (disk==1)):
       MW_rcm, MW_vcm, LMC_rcm, LMC_vcm = orbit(path, snap_name, initial_snap, final_snap,
                                                 NMW_particles, delta, lmc=False, disk=True)

   if ((lmc==0)& (disk==0)):
       MW_rcm, MW_vcm, LMC_rcm, LMC_vcm = orbit(path, snap_name, initial_snap, final_snap,
                                                 NMW_particles, delta, lmc=False, disk=False)

   write_orbit(out_name, MW_rcm, MW_vcm, LMC_rcm, LMC_vcm)
