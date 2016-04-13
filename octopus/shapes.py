import numpy as np
from scipy import linalg

"""
To-Do:

1. Rotation
2. Weight function
3. Implement volumes
"""

def shells(x, y, z, width, r, q, s):
    r_shell = np.sqrt(x**2.0 +y**2.0/q**2.0 +  z**2.0/s**2.0)
    index_shell = np.where((r_shell<r) & (r_shell>(r-width)))[0]
    x_shell = x[index_shell]
    y_shell = y[index_shell]
    z_shell = z[index_shell]
    return x_shell, y_shell, z_shell

def volumes(x, y, z, r, q, s):
    r_vol = np.sqrt(x**2.0 +y**2.0/q**2.0 +  z**2.0/s**2.0)
    index_vol = np.where(r_vol<r)[0]
    x_vol = x[index_vol]
    y_vol = y[index_vol]
    z_vol = z[index_vol]
    return x_vol, y_vol, z_vol

#Computing the shape tensor
def shape_tensor(x, y, z):
    N = len(x)
    XYZ = np.array([x, y, z])
    shape_T = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            XX = np.zeros(N)
            for n in range(N):
                XX[n] = XYZ[i,n] * XYZ[j,n]
            shape_T[i][j] = sum(XX) / N
    return shape_T

#Computing the axis ratios from the
#eigenvalues of the Shape Tensor
def axis_ratios(shape_T):
    eival, evec = linalg.eig(shape_T)
    oeival = np.sort(eival)
    a = oeival[2]
    b = oeival[1]
    c = oeival[0]
    s = np.sqrt(c/a)
    q = np.sqrt(b/a)
    return evec, s, q

def iterate(x, y, z, r, dr, tol):
    s_i = 1.0 #first guess of shape
    q_i = 1.0
    x_s, y_s, z_s = shells(x, y, z, dr, r, q_i, s_i)
    s_tensor = shape_tensor(x_s, y_s, z_s)
    rot_i, s, q = axis_ratios(s_tensor)
    while ((abs(s-s_i)>tol) & (abs(q-q_i)>tol)):
        s_i, q_i = s, q
        x_s, y_s, z_s = shells(x, y, z, dr, r, q_i, s_i)
        s_tensor = shape_tensor(x_s, y_s, z_s)
        rot, s, q = axis_ratios(s_tensor)
    return s, q
