"""
Code to select a particle orbit from a N-body simulation.



"""

import numpy as np
from pygadgetreader import readsnap

def index_particle(path, snap_name, R_particle, V_particle, time):
    """
    Finds the index of a particle in a given snapshot given it's
    velocity magnitude and position magnitude.

    Input:
    ------

    path: path to snapshot
    snap_name: snapshot base name.
    R_particle: Galactocentric distance of the particle
    V_particle: Galactocentric velocity of the particle
    time: Number of the snapshot.
    """
    # reading snapshot
    pos = readsnap(path + snap_name + '_{:03d}.hdf5'.format(time),'pos', 'dm')
    vel = readsnap(path + snap_name + '_{:03d}.hdf5'.format(time),'vel', 'dm')
    indexes = readsnap(path + snap_name + '_{:03d}.hdf5'.format(time),'pid', 'dm')

    # Computing the radius of all particles
    R = np.sqrt(pos[:,0]**2.0 + pos[:,1]**2.0 + pos[:,2]**2.0)

    # Selecting the desired radius range
    index_r = np.where(np.abs(R-R_particle) <1)[0]
    vel_cut = vel[index_r]
    pos_cut = pos[index_r]
    index_ir = indexes[index_r]

    # Selecting range of velocities
    V = np.sqrt(vel_cut[:,0]**2.0 + vel_cut[:,1]**2.0 + vel_cut[:,2]**2.0)
    index_v = np.where(np.abs(V-V_particle) == min(np.abs(V-V_particle)))[0]
    if len(index_v)==0:
        print('Error: There are not particles at {} kpc with a velocity of {}km/s\n'.format(R_particle, V_particle))
        print('The particles at this distance have the following range of velocities:', min(vel_cut), max(vel_cut))
        return 0
    else:
        print('Initial position (kpc):', pos_cut[index_v])
        print('Initial velocity (km/s):', vel_cut[index_v])
        return index_ir[index_v]

def N_body_orbit(path, snap_name, snap_n, pid):
    """
    Returns the positions and velocity of a particle given it's index
    """
    pos = readsnap(path + snap_name + '_{:03d}.hdf5'.format(snap_n),'pos', 'dm')
    vel = readsnap(path + snap_name + '_{:03d}.hdf5'.format(snap_n),'vel', 'dm')
    LMCMW_pid = readsnap(path + snap_name +'_{:03d}.hdf5'.format(snap_n), 'pid', 'dm')
    index_id = np.where(LMCMW_pid == pid)[0]
    return pos[index_id], vel[index_id]

def particle_orbit(path, snap_name, R, V, t, t_i, dt):
    """
    Finds a particle at a given position and
    velocity in a snapshot of a N-body simulation and returns
    its orbit in a given time reading all the snapshots.

    Parameters:
    -----------
    path: path to snapshots folder.
    snap_name: snapshots base name.
    R: Distance to a particle from a (0,0,0) reference point.
    V: Velocity of the particle at R.
    t: time of the orbit in Gyrs.
    t_i: Initial time of the orbit.
    dt: Time interval between snapshots.
    Returns:
    --------
    An array with the positions and velocities of the particle at all
    the snapshots. (2, N_snaps, 3)
    """

    N_i = int(t_i/dt)
    N_snaps = int(t/dt)
    particle_pos = np.zeros((N_snaps,3))
    particle_vel = np.zeros((N_snaps,3))
    pid = index_particle(path, snap_name, R, V, N_i)
    for i in range(N_i, N_snaps+N_i):
        particle_pos[i-N_i], particle_vel[i-N_i] = N_body_orbit(path, snap_name, i, pid)
    return particle_pos, particle_vel
