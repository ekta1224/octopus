# The output units are as folows:
# M = [Msun]
# rho = [Msun / kpc3]
# potential = 
# Circular velocity = [km/s]
# acceleration = [kpc/Gyr2]
#########################################################


import numpy as np
from astropy import constants
from astropy import units
from cosmotools import *


K = constants.k_B
K = K.to(units.Msun * units.kpc**2 / (units.s**2 * units.Kelvin))

#++++++++++++++++ PLUMMER ++++++++++++++++++++++++++++++++++++++

def mass_plummer(a, r, M):
    M = M*units.Msun
    r = r*units.kpc
    a = a*units.kpc
    Mass = M*r**3 / (a**2 + r**2)**(3/2.)
    return Mass.value

def dens_plummer(a, r, M):
    M = M*units.Msun
    r = r*units.kpc
    a = a*units.kpc
    rho = 3*M / (4 *np.pi * a**3) * (1 + r**2/a**2)**(-5/2)
    return rho.value

def pot_plummer(a, r, M):
    M = M*units.Msun
    r = r*units.kpc
    a = a*units.kpc
    phi =  - G*M / np.sqrt(r**2 + a**2)
    return phi.value

def vc_plummer(a, r, M):
    a = a*units.kpc
    M = M*units.Msun
    r = r*units.kpc
    vc = np.sqrt(G*M*( r**2/(r**2 + a**2)**(3/2.)))
    vc = vc.to(units.km / units.s)
    return vc.value

def a_plummer(a, x, y, z, M):
    a = a*units.kpc
    M = M*units.Msun
    x = x * units.kpc
    y = y * units.kpc
    z = z * units.kpc
    r = np.sqrt(x**2 + y**2 + z**2)
    Ax = - x *G * M / (r**2 + a**2)**(3/2.0)
    Ay = - y *G * M / (r**2 + a**2)**(3/2.0)
    Az = - z *G * M/ (r**2 + a**2)**(3/2.0)
    Ax = Ax.to(units.kpc / units.Gyr**2)
    Ay = Ay.to(units.kpc / units.Gyr**2)
    Az = Az.to(units.kpc / units.Gyr**2)
    return Ax.value, Ay.value, Az.value
	
#++++++++++++++++ HERNQUIST ++++++++++++++++++++++++++++

def pot_hernquist(a, r, M):
    a = a * units.kpc
    r = r * units.kpc
    M = M * units.Msun
    phi = -G*M / (r+a)
    return phi.value

def dens_hernquist(a, r, M):
    a = a * units.kpc
    r = r * units.kpc
    M = M * units.Msun
    rho = M / (2. * np.pi) * a / (r*(r+a)**3.)
    return rho.value

def mass_hernquist(a, r, M):
    a = a*  units.kpc
    r = r * units.kpc
    M = M * units.Msun
    Mass = M * r**2. / (r+a)**2.
    return Mass.value

def vc_hernquist(a, x, y, z, M):
    x = x * units.kpc
    y = y * units.kpc
    z = z * units.kpc
    a = a*units.kpc
    r = np.sqrt(x**2. + y**2. + z**2.)
    M = M * units.Msun
    vc = np.sqrt(G*M*r/(r+a)**2)
    vc = vc.to(units.km / units.s)
    return vc.value

def a_hernquist(a, x, y, z, M):
    a = a * units.kpc
    x = x * units.kpc
    y = y * units.kpc
    z = z * units.kpc
    r = np.sqrt(x**2 + y**2 + z**2)
    M = M * units.Msun
    Ax =  - 1.0 * x * G * M / ( r * (r + a)**2)
    Ay =  - 1.0 * y * G * M / ( r * (r + a)**2)
    Az =  - 1.0 * z * G * M / ( r * (r + a)**2)
    Ax = Ax.to(units.kpc / units.Gyr**2)
    Ay = Ay.to(units.kpc / units.Gyr**2)
    Az = Az.to(units.kpc / units.Gyr**2)
    return Ax.value, Ay.value, Az.value

#+++++++++++++++++++ SIS (Singular Isothermal Sphere) ++++++++++++++++++++

def dens_sis(a, r, v):
    a = a * units.kpc
    r = r * units.kpc
    v = v * units.km / units.s
    v = v.to(units.kpc / units.s)
    rho = v**2 / (4 * np.pi * G * (r**2 + a**2))
    return rho.value

def mass_sis(a, r, v):
    a = a * units.kpc
    r = r * units.kpc
    v = v * units.km / units.s
    v = v.to(units.kpc / units.s)
    M = v**2 * r/G
    return M.value

def pot_sis(a, r, v):
    a = a * units.kpc
    r = r * units.kpc
    v = v * units.km / units.s
    v = v.to(units.kpc / units.s)
    phi = v**2 * np.log(r.value + a.value)
    return phi.value

def vc_sis(a, r, v):
    a = a * units.kpc
    r = r * units.kpc
    v = v * units.km / units.s
    v = v.to(units.kpc / units.s)
    V = v * np.sqrt(r + a) / np.sqrt(r)
    V = V.to(units.km / units.s )
    return V.value

def a_sis(a, x, y, z, v):
    a = a * units.kpc
    x = x * units.kpc
    y = y * units.kpc
    z = z * units.kpc
    r = np.sqrt(x**2 + y**2 + z**2)
    v = v * units.km / units.s
    v = v.to(units.kpc / units.s)
    Ax = -v**2 * x / (r * (r + a))
    Ay = -v**2 * y / (r * (r + a))
    Az = -v**2 * z / (r * (r + a))
    Ax = Ax.to(units.kpc/units.Gyr**2)
    Ay = Ay.to(units.kpc/units.Gyr**2)
    Az = Az.to(units.kpc/units.Gyr**2)
    return Ax.value, Ay.value, Az.value


#+++++++++++++++++++++++ Miyamoto-Nagai +++++++++++++++++++++++++++

def pot_mn(a, b, x, y, z, M):
    x = x * units.kpc
    y = y * units.kpc
    z = z * units.kpc
    a = a * units.kpc
    b = b * units.kpc
    R = np.sqrt(x**2 + y**2)
    M = M * units.Msun
    phi = - G*M / (np.sqrt(R**2 + ( a + np.sqrt( z**2 + b**2 ))**2 ) )
    return phi.value

def dens_mn(a, b, x, y, z, M):
    x = x * units.kpc
    y = y * units.kpc
    z = z * units.kpc
    a = a * units.kpc
    b = b * units.kpc
    M = M * units.Msun
    R = np.sqrt(x**2 + y**2)
    rho = (b**2 * M / (4*np.pi)) * (a*R**2 + ( a + 3*(np.sqrt(z**2 + b**2)))*( a + np.sqrt(z**2 + b**2))**2 ) /( ( (R**2 + (a + np.sqrt(z**2 + b**2))**2)**(5./2.) * (z**2 + b**2)**(3./2.)) )
    return rho.value

def vc_mn(a, b, x, y, z,  M):
    x = x * units.kpc
    y = y * units.kpc
    z = z * units.kpc
    a = a * units.kpc
    b = b * units.kpc
    R = np.sqrt(x**2 + y**2)
    M = M * units.Msun
    factor = R**2 + (a + np.sqrt(z**2 + b**2))**2
    vc = np.sqrt(G * M * R**2 / factor**(3.0/2.0))
    vc = vc.to(units.km / units.s)
    return vc.value

def mass_mn(a, b, x, y, z, M):
    x = x * units.kpc
    y = y * units.kpc
    z = z * units.kpc
    a = a * units.kpc
    b = b * units.kpc
    R = np.sqrt(x**2 + y**2)
    M = M * units.Msun
    v = vc_mn(a.value, b.value, z.value, R.value, M.value)
    v = v.to(units.km/units.s)
    mass = v**2 * R / G
    return mass.value

def a_mn(a, b, x, y, z, M):
    x = x * units.kpc
    y = y * units.kpc
    z = z * units.kpc
    a = a * units.kpc
    b = b * units.kpc
    R = np.sqrt(x**2 + y**2)
    M = M * units.Msun
    Ax = -  x * G * M / (R**2 + (a + np.sqrt( z**2 + b**2))**2)**(3.0/2.0)
    Ay = -  y * G * M / (R**2 + (a + np.sqrt( z**2 + b**2))**2)**(3.0/2.0)
    Az = -  z * G * M * (a + np.sqrt(z**2 + b**2)) / ((R**2 + (a + np.sqrt(z**2 + b**2))**2)**(3.0/2.0) * np.sqrt(z**2 + b**2))
    Ax = Ax.to(units.kpc/units.Gyr**2)
    Ay = Ay.to(units.kpc/units.Gyr**2)
    Az = Az.to(units.kpc/units.Gyr**2)
    return Ax.value, Ay.value, Az.value

#+++++++++++++++++++++++++ NFW +++++++++++++++++++++++++++

def pot_NFW(c, x, y, z, M):
    x = x*units.kpc
    y = y*units.kpc
    z = z*units.kpc
    r = np.sqrt(x**2 + y**2 + z**2)
    Rvir = rvir(M, 0) # here we are working at z=0
    a = Rvir / c
    M = M * units.Msun
    f = np.log(1.0 + Rvir/a) - (Rvir/a / (1.0 + Rvir/a))
    phi = -G * M * np.log(1 + r/a) / (r * f)
    return phi.value

def dens_NFW(c, x, y, z, M):
    x = x*units.kpc
    y = y*units.kpc
    z = z*units.kpc
    r = np.sqrt(x**2 + y**2 + z**2)
    Rvir = rvir(M, 0) # here we are working at z=0
    a = Rvir / c
    M = M * units.Msun
    f = np.log(1.0 + Rvir/a) - (Rvir/a / (1.0 + Rvir/a))
    rho = M / ((4.0 * np.pi * a**3.0 * f) * (r / a) * (1.0 + (r/a))**2.0)
    return rho.value

def dens_NFWnRvir(c, x, y, z, M, Rv):
    x = x*units.kpc
    y = y*units.kpc
    z = z*units.kpc
    r = np.sqrt(x**2 + y**2 + z**2)
    Rvir = Rv * units.kpc# here we are working at z=0
    a = Rvir / c
    M = M * units.Msun
    f = np.log(1.0 + Rvir/a) - (Rvir/a / (1.0 + Rvir/a))
    rho = M / ((4.0 * np.pi * a**3.0 * f) * (r / a) * (1.0 + (r/a))**2.0)
    rho = rho.to(units.Msun / units.kpc**3)
    return rho.value


def vc_NFW(c, x, y, z, M):
    x = x*units.kpc
    y = y*units.kpc
    z = z*units.kpc
    r = np.sqrt(x**2 + y**2 + z**2)
    Rvir = rvir(M, 0) # here we are working at z=0
    M = M * units.Msun
    a = Rvir / c
    f = np.log(1.0 + Rvir/a) - (Rvir/a / (1.0 + Rvir/a))
    up = G * M * (np.log(1 + r/a) - r/(r+a)) / f
    vc = np.sqrt(up / r)
    vc = vc.to(units.km / units.s)
    return vc.value

def mass_NFW(c, x, y, z, M):
    x = x*units.kpc
    y = y*units.kpc
    z = z*units.kpc
    Rvir = rvir(M, 0) # here we are working at z=0
    a = Rvir / c
    M = M * units.Msun
    r = np.sqrt(x**2 + y**2 + z**2)
    f = np.log(1.0 + c) - (c / (1.0 + c))
    mass = M * (np.log(1 + r/a) - r/(a+r)) / f
    return mass.value

def mass_NFW200(c200, x, y, z, M, R200):
    x = x*units.kpc
    y = y*units.kpc
    z = z*units.kpc
    #R200 = r200(M, 0) # here we are working at z=0
    R200 = R200*units.kpc
    a = R200 / c200
    M = M * units.Msun
    r = np.sqrt(x**2 + y**2 + z**2)
    f = np.log(1.0 + c200) - (c200 / (1.0 + c200))
    mass = M * (np.log(1 + r/a) - r/(a+r)) / f
    return mass.value

def mass_NFWnRvir(c, x, y, z, M, Rv):
    x = x*units.kpc
    y = y*units.kpc
    z = z*units.kpc
    Rvir = Rv * units.kpc # here we are working at z=0
    a = Rvir / c
    M = M * units.Msun
    r = np.sqrt(x**2 + y**2 + z**2)
    f = np.log(1.0 + c) - (c / (1.0 + c))
    mass = M * (np.log(1 + r/a) - r/(a+r)) / f
    return mass.value

def a_NFW(c, x, y, z, M):
    x = x*units.kpc
    y = y*units.kpc
    z = z*units.kpc
    #M = M * units.Msun
    Rvir = rvir(M, 0) # here we are working at z=0
    M = M * units.Msun
    a = Rvir / c
    r = np.sqrt(x**2 + y**2 + z**2)
    f = np.log(1.0 + Rvir/a) - (Rvir/a / (1.0 + Rvir/a))
    ax = G * M / r**2 * (r/(r+a) - np.log(1 + r/a)) * x / r / f
    ay = G * M / r**2 * (r/(r+a) - np.log(1 + r/a)) * y / r / f
    az = G * M / r**2 * (r/(r+a) - np.log(1 + r/a)) * z / r / f
    ax = ax.to(units.kpc/units.Gyr**2)
    ay = ay.to(units.kpc/units.Gyr**2)
    az = az.to(units.kpc/units.Gyr**2)
    return ax.value, ay.value, az.value


def a_NFWnRvir(c, x, y, z, M, Rv):
    x = x*units.kpc
    y = y*units.kpc
    z = z*units.kpc
    Rvir = Rv * units.kpc
    M = M * units.Msun
    a = Rvir / c
    r = np.sqrt(x**2 + y**2 + z**2)
    f = np.log(1.0 + Rvir/a) - (Rvir/a / (1.0 + Rvir/a))
    ax = G * M / r**2 * (r/(r+a) - np.log(1 + r/a)) * x / r / f
    ay = G * M / r**2 * (r/(r+a) - np.log(1 + r/a)) * y / r / f
    az = G * M / r**2 * (r/(r+a) - np.log(1 + r/a)) * z / r / f
    ax = ax.to(units.kpc/units.Gyr**2)
    ay = ay.to(units.kpc/units.Gyr**2)
    az = az.to(units.kpc/units.Gyr**2)
    return ax.value, ay.value, az.value


#+++++++++++++++++++++++++++++++++++++++++++ Logarithmic Profile +++++++++++++++++++++++

def pot_log(Rc, q, z, R, v):
    z = z * units.kpc
    Rc = Rc * units.kpc
    v = v * units.km / units.s
    v = v.to(units.kpc / units.s)
    R = R * units.kpc
    phi = 0.5 * v**2 * np.log(Rc.value**2 + R.value**2 + z.value**2/q**2)
    return phi.value

def dens_log(Rc, q, z, R, v):
    z = z * units.kpc
    Rc = Rc * units.kpc
    v = v * units.km / units.s
    v = v.to(units.kpc / units.s)
    R = R * units.kpc
    rho = (v**2 / (4*np.pi*G*q**2)) * (((2*q**2 + 1)*Rc**2 + R**2 + (2 - q**-2)*z**2) / (Rc**2 + R**2 + (z**2 / q**2))**2)
    return rho.value

def vc_log(Rc, q, z, R, v):
    z = z * units.kpc
    Rc = Rc * units.kpc
    v = v * units.km / units.s
    v = v.to(units.kpc / units.s)
    R = R * units.kpc
    vc = v * R / np.sqrt(Rc**2 + R**2 + z**2/q**2)
    vc = vc.to(units.km/units.s)
    return vc.value

def mass_log(Rc, q, z, R, v):
    z = z * units.kpc
    Rc = Rc * units.kpc
    v = v * units.km / units.s
    v = v.to(units.kpc / units.s)
    R = R * units.kpc
    M = v**2 * R**3 / (G * (Rc**2 + R**2 + z**2/q**2))
    return M.value

def a_log(Rc, q, z, R, v):
    z = z * units.kpc
    Rc = Rc * units.kpc
    v = v * units.km / units.s
    v = v.to(units.kpc / units.s)
    R = R * units.kpc
    factor = Rc**2 + R**2 + z**2 / q**2
    aR = - v**2 * R / factor 
    az = - (v**2 * z/q**2) / factor
    aR = aR.to(units.kpc / units.Gyr**2)
    az = az.to(units.kpc / units.Gyr**2)
    return aR.value, az.value

#++++++++++++++++++++ Triaxial LMJ++++++++++++++++++++

def constants(q1, q2, qz, phi):
       C1 = (np.cos(phi)**2 / q1**2)  + (np.sin(phi)**2 / q2**2)
       C2 = (np.cos(phi)**2 / q2**2)  + (np.sin(phi)**2 / q1**2)
       C3 = 2*np.sin(phi)*np.cos(phi)*(1/q1**2 - 1/q2**2)
       return C1, C2, C3

def pot_LMJ(r_h, q1, q2, qz, phi, x, y, z, v):
       r_h = r_h * units.kpc
       z = z * units.kpc
       x = x * units.kpc
       y = y * units.kpc
       v = v * units.km / units.s
       v = v.to(units.kpc / units.s)
       C1, C2, C3 = constants(q1, q2, qz, phi)
       phi = v**2 * np.log(C1*x**2 + C2*y**2 + C3*x*y + (z/qz)**2 + r_h**2)
       return phi.value

def vc_LMJ(r_h, q1, q2, qz, phi, x, y, z, v):
       r_h = r_h * units.kpc
       z = z * units.kpc
       x = x * units.kpc
       y = y * units.kpc
       v = v * units.km / units.s
       v = v.to(units.kpc / units.s)
       C1, C2, C3 = constants(q1, q2, qz, phi)
       factor = (C1*x**2 + C2*y**2 + C3*x*y + (z**2/qz**2 + r_h**2))
       r = np.sqrt(x**2 + y**2 + z**2)
       vc = v * np.sqrt(r*np.sqrt((2*C1*x + C3*y)**2 + (2*C2*y + C3*x)**2 + (2*z/qz**2)**2) / factor)
       vc = vc.to(units.km / units.s)
       return vc.value

def a_LMJ(r_h, q1, q2, qz, phi, x, y, z, v):
       r_h = r_h * units.kpc
       z = z * units.kpc
       x = x * units.kpc
       y = y * units.kpc
       v = v * units.km / units.s
       v = v.to(units*kpc / units.s)
       C1, C2, C3 = constants(q1, q2, qz, phi)
       factor = (C1*x**2 + C2*y**2 + C3*x*y + (z**2/qz**2 + r_h**2))
       ax = -v**2 * (2*C1*x + C3*y) / factor
       ay = -v**2 * (2*C2*y + C3*x) / factor
       az = -v**2 * (2*z / qz**2) / factor
       ax = ax.to(units.kpc / units.Gyr**2)
       ay = ay.to(units.kpc / units.Gyr**2)
       az = az.to(units.kpc / units.Gyr**2)
       return ax.value, ay.value, az.value

def mass_LMJ(r_h, q1, q2, qz, phi, x, y, z, v):
       r_h = r_h * units.kpc
       z = z * units.kpc 
       x = x * units.kpc
       y = y * units.kpc
       v = v * units.km / units.s
       v = v.to(units*kpc / units.s)
       C1, C2, C3 = constants(q1, q2, qz, phi)
       r = np.sqrt(x**2 + y**2 + z**2)
       factor1 = 2*C1*x + C3*y
       factor2 = 2*C2*y + C3*x
       factor3 = 2*z/qz**2
       factor = (C1*x**2 + C2*y**2 + C3*x*y + (z**2/qz**2 + r_h**2))
       M = v**2 * r**2 * (factor1**2 + factor2**2 + factor3**2) / (G *
factor)
       return M.value

#+++++++++++++++++++++++++Vera-Ciro-Helmi++++++++++++++++++++++++++++++++++++++++++++++

def pot_VCH(d, q1, q2, q3, qz, phi, ra, x, y, z, v):
    c1, c2, c3 = constants(q1, q2, q3, phi)
    rA = np.sqrt(x**2 + y**2 + z**2/qz**2)
    rT = np.sqrt(c1*x**2 + c2*y**2 + c3*x*y + z**2/q3**2)
    r = (ra + rT)*rA / (ra + rA)
    pot = v**2 * log(r**2 + d**2)
    return pot

def vc_VCH():
    vc = np.sqrt((2 * r**2 * v**2 / (r**2 + d**2)) * np.sqrt(drdx**2 + drdy**2 + drdz**2))
