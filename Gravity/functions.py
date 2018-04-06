import numpy as np
import math

"""
v 1.0
 Gravity modelling
   with point mass and prism models

 Author: I. Vasconcelos  2016
 Translated to python: S. Schneider 2018
"""


def mpoinCart(dx, dy, dz, mass):

    G = 6.67259E-11  # gravitational constant    [m^2/(kg*s^2)]

    l = 1./(np.sqrt(dx**2 + dy**2 + dz**2))
    l3 = l*l*l
    l5 = l*l*l3

    Vxx = G*mass*(-l3 + 3*dx*dx*l5)
    Vxy = G*mass*(3*dx*dy*l5)
    Vxz = G*mass*(3*dx*dz*l5)
    Vyy = G*mass*(-l3 + 3*dy*dy*l5)
    Vyz = G*mass*(3*dy*dz*l5)
    Vzz = G*mass*(-l3 + 3*dz*dz*l5)

    Vz = -G*mass*(dz*l3)
    P = G*mass*l

    return Vxx, Vxy, Vxz, Vyy, Vyz, Vzz, P, Vz


def mPrismCart(x, y, z, mass, xi=None, yi=None, zi=None):
    G = 6.67259E-11
    if not xi:
        xi = np.zeros(2)
        xi[0] = 1E2
        xi[1] = -1E2

    if not yi:
        yj = np.zeros(2)
        yj[0] = 1E2
        yj[1] = -1E2

    if not zi:
        zk = np.zeros(2)
        zk[0] = 1E2
        zk[1] = -1E2

    V = (xi[0]-xi[1]) * (yj[0]-yj[1]) * (zk[0]-zk[1])
    Rho = mass/V
    P = 0

    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                wijk = np.sqrt((x-xi[i])**2+(y-yj[j])**2+(z-zk[k])**2)

                V_tmp = (
                         (x-xi[i])*(y-yj[j]) *
                         np.log(abs((z-zk[k]+wijk)/np.sqrt((x-xi[i])**2
                                + (y-yj[j])**2)))
                         + (y-yj[j])*(z-zk[k]) *

                         np.log(abs((x-xi[i]+wijk)/np.sqrt((y-yj[j])**2
                                + (z-zk[k])**2)))
                         + (z-zk[k])*(x-xi[i]) *

                         np.log(abs((y-yj[j]+wijk)/np.sqrt((z-zk[k])**2
                                + (x-xi[i])**2)))

                         - 1/2*(
                                (x-xi[i])**2 *
                                math.atan2((y-yj[j])*(z-zk[k]), (x-xi[i])*wijk)
                                +
                                (y-yj[j])**2 *
                                math.atan2((z-zk[k])*(x-xi[i]), (y-yj[j])*wijk)
                                +
                                (z-zk[k])**2 *
                                math.atan2((x-xi[i])*(y-yj[j]), (z-zk[k])*wijk)
                                )
                         )
                P = P + G*Rho * (-1)**(i+1 + j+1 + k+1) * V_tmp

    ########################
    deltax1 = x-xi[0]
    deltax2 = x-xi[1]
    deltay1 = y-yj[0]
    deltay2 = y-yj[1]
    deltaz1 = z-zk[0]
    deltaz2 = z-zk[1]
    ########################
    res = 0

    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                r = np.sqrt((x-xi[i])**2 + (y-yj[j])**2 + (z-zk[k])**2)
                kernel = (
                          (x-xi[i]) * np.log((y-yj[j]) + r) +
                          (y-yj[j]) * np.log((x-xi[i]) + r) -
                          (z-zk[k]) * math.atan2((x-xi[i])*(y-yj[j]),
                                                 (z-zk[k])*r)
                         )
                res += (-1)**(i+1 + j+1 + k+1)*kernel

    Vz = res*G*Rho

    ########################
    res = 0
    # /* Evaluate the integration limits */
    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res + 1*math.atan2(deltay1*deltaz1, deltax1*r)
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res - 1*math.atan2(deltay1*deltaz1, deltax2*r)
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res - 1*math.atan2(deltay2*deltaz1, deltax1*r)
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res + 1*math.atan2(deltay2*deltaz1, deltax2*r)
    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res - 1*math.atan2(deltay1*deltaz2, deltax1*r)
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res + 1*math.atan2(deltay1*deltaz2, deltax2*r)
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res + 1*math.atan2(deltay2*deltaz2, deltax1*r)
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res - 1*math.atan2(deltay2*deltaz2, deltax2*r)
    # Now all that is left is to multiply res by the gravitational constant
    # and density and convert it to Eotvos units */
    Vxx = res*G*Rho  # *= G*SI2EOTVOS*prism.density
    ########################

    res = 0
    # /* Evaluate the integration limits */
    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res + 1*(-1*np.log(deltaz1 + r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res + -1*(-1*np.log(deltaz1 + r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res + -1*(-1*np.log(deltaz1 + r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res + 1*(-1*np.log(deltaz1 + r))
    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res + -1*(-1*np.log(deltaz2 + r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res + 1*(-1*np.log(deltaz2 + r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res + 1*(-1*np.log(deltaz2 + r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res + -1*(-1*np.log(deltaz2 + r))
    # /* Now all that is left is to multiply res by the gravitational
    # constant and density and convert it to Eotvos units */
    ########################

    Vxy = res*G*Rho  # *= G*SI2EOTVOS*prism.density

    #############################
    res = 0
    # /* Evaluate the integration limits */
    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1)
    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1)

    res = res + 1*(-1*np.log(deltaz1 + r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res + -1*(-1*np.log(deltaz1 + r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res + -1*(-1*np.log(deltaz1 + r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res + 1*(-1*np.log(deltaz1 + r))
    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res + -1*(-1*np.log(deltaz2 + r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res + 1*(-1*np.log(deltaz2 + r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res + 1*(-1*np.log(deltaz2 + r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res + -1*(-1*np.log(deltaz2 + r))
    # /* Now all that is left is to multiply res by the gravitational
    # constant and
    # density and convert it to Eotvos units */
    Vxy = res*G*Rho  # *= G*SI2EOTVOS*prism.density
    #############################
    res = 0

    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res + 1*(-1*np.log(deltay1 + r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res + -1*(-1*np.log(deltay1 + r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res + -1*(-1*np.log(deltay2 + r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res + 1*(-1*np.log(deltay2 + r))
    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res + -1*(-1*np.log(deltay1 + r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res + 1*(-1*np.log(deltay1 + r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res + 1*(-1*np.log(deltay2 + r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res + -1*(-1*np.log(deltay2 + r))

    Vxz = res*G*Rho  # G*SI2EOTVOS*prism.density
    #############################

    res = 0

    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res + 1*(math.atan2(deltaz1*deltax1, deltay1*r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res - 1*(math.atan2(deltaz1*deltax2, deltay1*r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res - 1*(math.atan2(deltaz1*deltax1, deltay2*r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res + 1*(math.atan2(deltaz1*deltax2, deltay2*r))
    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res - 1*(math.atan2(deltaz2*deltax1, deltay1*r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res + 1*(math.atan2(deltaz2*deltax2, deltay1*r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res + 1*(math.atan2(deltaz2*deltax1, deltay2*r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res - 1*(math.atan2(deltaz2*deltax2, deltay2*r))

    Vyy = res*G*Rho
    #############################
    res = 0

    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res + 1*(-1*np.log(deltax1 + r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res + -1*(-1*np.log(deltax2 + r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res + -1*(-1*np.log(deltax1 + r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res + 1*(-1*np.log(deltax2 + r))
    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res + -1*(-1*np.log(deltax1 + r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res + 1*(-1*np.log(deltax2 + r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res + 1*(-1*np.log(deltax1 + r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res + -1*(-1*np.log(deltax2 + r))

    Vyz = res*G*Rho
    #############################
    res = 0

    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res + 1*(math.atan2(deltax1*deltay1, deltaz1*r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz1*deltaz1)
    res = res - 1*(math.atan2(deltax2*deltay1, deltaz1*r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res - 1*(math.atan2(deltax1*deltay2, deltaz1*r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz1*deltaz1)
    res = res + 1*(math.atan2(deltax2*deltay2, deltaz1*r))
    r = np.sqrt(deltax1*deltax1 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res - 1*(math.atan2(deltax1*deltay1, deltaz2*r))
    r = np.sqrt(deltax2*deltax2 + deltay1*deltay1 + deltaz2*deltaz2)
    res = res + 1*(math.atan2(deltax2*deltay1, deltaz2*r))
    r = np.sqrt(deltax1*deltax1 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res + 1*(math.atan2(deltax1*deltay2, deltaz2*r))
    r = np.sqrt(deltax2*deltax2 + deltay2*deltay2 + deltaz2*deltaz2)
    res = res - 1*(math.atan2(deltax2*deltay2, deltaz2*r))

    Vzz = res*G*Rho
    #############################

    return Vxx, Vxy, Vxz, Vyy, Vyz, Vzz, P, Vz
