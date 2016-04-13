#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pygadgetreader import *
import sys
import argparse

if len(sys.argv) != 8:
    print 'Usage: python orbit_cm.py snap_base_name inital_snap_number'\
           'final_snap_number path2file out?name  #DMhost #DMsat'
    print 'Ex: python orbit_cm.py snap 0 50 pat2file out_name Nhost Nsat'
    sys.exit(0)


# global variables to read the snapshots
snap = str(sys.argv[1])
# Initial and final snapshot number
i_n = int(sys.argv[2])
i_f = int(sys.argv[3])
#Output Name
out_name = str(sys.argv[5])
#Number of particles of the host
Nhost = int(sys.argv[6])
#Number of particles of the sat
Nsat = int(sys.argv[7])
path = str(sys.argv[4]) #'../../data/LMCMW/MW1LMC4/a1/'

# Number of Snapshots
N_snaps = (i_f - i_n) + 1

D = 0.0333 # precision of the CM computation in Kpc.
V_radius = 2.0 # Radius of disk particles to compute the CM
#Position and velocity arrays for the host and the satellite
X = np.zeros(N_snaps)
Y = np.zeros(N_snaps)
Z = np.zeros(N_snaps)

VX = np.zeros(N_snaps)
VY = np.zeros(N_snaps)
VZ = np.zeros(N_snaps)

Xsat = np.zeros(N_snaps)
Ysat = np.zeros(N_snaps)
Zsat = np.zeros(N_snaps)

VXsat = np.zeros(N_snaps)
VYsat = np.zeros(N_snaps)
VZsat = np.zeros(N_snaps)

#Galactocentric distance and time arrays
Rgal = np.zeros(N_snaps)
Vgal = np.zeros(N_snaps)
time = np.zeros(N_snaps)

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
def CM(x, y, z, vx, vy, vz, delta):
    N = len(x)
    xCM = sum(x)/N
    yCM = sum(y)/N
    zCM = sum(z)/N

    xCM_new = xCM
    yCM_new = yCM
    zCM_new = zCM

    xCM = 0.0
    yCM = 0.0
    zCM = 0.0

    vxCM_new = sum(vx)/N
    vyCM_new = sum(vy)/N
    vzCM_new = sum(vz)/N

    R1 = np.sqrt((x - xCM_new)**2 + (y - yCM_new)**2 + (z - zCM_new)**2)
    i=0
    while (np.sqrt((xCM_new-xCM)**2 + (yCM_new-yCM)**2 +(zCM_new-zCM)**2) > delta):
        xCM = xCM_new
        yCM = yCM_new
        zCM = zCM_new
        Rcm = np.sqrt(xCM**2 + yCM**2 + zCM**2)
        R = np.sqrt((x - xCM)**2 + (y - yCM)**2 + (z - zCM)**2)
        Rmax = max(R)
        index = np.where(R<Rmax/1.3)[0]
        x = x[index]
        y = y[index]
        z = z[index]
        vx = vx[index]
        vy = vy[index]
        vz = vz[index]
        N = len(x)
        i+=1
        xCM_new = sum(x)/N
        yCM_new = sum(y)/N
        zCM_new = sum(z)/N
        vxCM_new = sum(vx)/N
        vyCM_new = sum(vy)/N
        vzCM_new = sum(vz)/N
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

for i in range(i_n, i_f + 1):
    print 'snapshot' + str(i)
    time[i-i_n] = readheader(path + snap + "_{:0>3d}".format(i),'time')
    positions = readsnap(path + snap + "_{:0>3d}".format(i),'pos', 'dm')
    velocities = readsnap(path + snap + "_{:0>3d}".format(i), 'vel', 'dm')
    particles_ids = readsnap(path + snap + "_{:0>3d}".format(i), 'pid', 'dm')
    potential = readsnap(path + snap + "_{:0>3d}".format(i), 'pot','disk')
    positions_d = readsnap(path + snap + "_{:0>3d}".format(i),'pos','disk')
    v_d = readsnap(path + snap + "_{:0>3d}".format(i),'vel','disk')

    ID = np.sort(particles_ids)
    # The first set of particles are from the host DM halo, the
    # second set are from the satellite DM halo, the limit is know by
    # the number of particles in the host halo.
    idcut = ID[Nhost-1]
    #index_mw = np.where(particles_ids<=idcut)
    index_LMC = np.where(particles_ids>idcut)

    x_mw = positions_d[:,0]
    y_mw = positions_d[:,1]
    z_mw = positions_d[:,2]
    #x_mw = positions[index_mw[0],0]
    #y_mw = positions[index_mw[0],1]
    #z_mw = positions[index_mw[0],2]
    x_lmc = positions[index_LMC[0],0]
    y_lmc = positions[index_LMC[0],1]
    z_lmc = positions[index_LMC[0],2]

    vx_mw = v_d[:,0]
    vy_mw = v_d[:,1]
    vz_mw = v_d[:,2]
    #vx_mw = velocities[index_mw[0],0]
    #vy_mw = velocities[index_mw[0],1]
    #vz_mw = velocities[index_mw[0],2]
    vx_lmc = velocities[index_LMC[0],0]
    vy_lmc = velocities[index_LMC[0],1]
    vz_lmc = velocities[index_LMC[0],2]

    #potmw = potential[index_mw]
    #potlmc = potential[index_LMC]

    X[i-i_n], Y[i-i_n], Z[i-i_n], VX[i-i_n], VY[i-i_n], VZ[i-i_n] = CM_disk_potential(x_mw, y_mw, z_mw, vx_mw, vy_mw, vz_mw, potential)
    Xsat[i-i_n], Ysat[i-i_n], Zsat[i-i_n], VXsat[i-i_n], VYsat[i-i_n], VZsat[i-i_n]= CM(x_lmc, y_lmc, z_lmc, vx_lmc, vy_lmc, vz_lmc, D)
    Rgal[i-i_n] = np.sqrt((X[i-i_n] - Xsat[i-i_n])**2 + (Y[i-i_n]-Ysat[i-i_n])**2 +(Z[i-i_n] - Zsat[i-i_n])**2)
    Vgal[i-i_n] = np.sqrt((VX[i-i_n] - VXsat[i-i_n])**2 + (VY[i-i_n]-VYsat[i-i_n])**2 + (VZ[i-i_n] - VZsat[i-i_n])**2)

print 'Writing the data'
f = open(out_name, 'w')
f.write("#Time(Gyrs) | Rgal(kpc) | Xsat[kpc] | Ysat[kpc] | Zsat[kpc] |Xhost[kpc] | Yhost[kpc] Zhost[kpc] |"\
        "Vgal | Vxsat | Vysat | Vzsat | Vxhost | Vyhost | Vzhost |\n")

for i in range(0, len(Rgal)):
    f.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n"%(time[i], Rgal[i], Xsat[i], Ysat[i],\
    Zsat[i], X[i], Y[i], Z[i], Vgal[i], VXsat[i], VYsat[i], VZsat[i], VX[i], VY[i], VZ[i]))
f.close()
