import numpy as np
import sys
from Gravity.functions import (mPrismCart, mpoinCart, mHollowSphere,
                               update_progress)
from Gravity.plotting import plot_gravity, plot_hollow_sphere


"""
v 1.0
 Gravity modelling
   with point mass and prism models

 Author: I. Vasconcelos  2016
 Translated to python: S. Schneider 2018
"""


def model_gravity(object='all', scal=1, plotting=True):
    """
    object: 'prisma', 'point' or 'all'
    """

    nstart = -1E3
    nend = 1E3
    nstep = 2E1
    x = scal * np.arange(nstart, nend+nstep, nstep) + 1E-3
    y = scal * np.arange(nstart, nend+nstep, nstep) + 1E-3
    z = (scal)**1.5*1.1E2

    mass = 7.1430e+012

    Vxx1 = np.zeros((x.size, y.size))
    Vxy1 = np.zeros((x.size, y.size))
    Vxz1 = np.zeros((x.size, y.size))
    Vyy1 = np.zeros((x.size, y.size))
    Vyz1 = np.zeros((x.size, y.size))
    Vzz1 = np.zeros((x.size, y.size))
    Vz1 = np.zeros((x.size, y.size))
    P1 = np.zeros((x.size, y.size))

    Vxx2 = np.zeros((x.size, y.size))
    Vxy2 = np.zeros((x.size, y.size))
    Vxz2 = np.zeros((x.size, y.size))
    Vyy2 = np.zeros((x.size, y.size))
    Vyz2 = np.zeros((x.size, y.size))
    Vzz2 = np.zeros((x.size, y.size))
    Vz2 = np.zeros((x.size, y.size))
    P2 = np.zeros((x.size, y.size))

    update_progress(0)
    for j, v in enumerate(y):
        for i, w in enumerate(x):
            if object in ['prisma', 'all']:
                V = mPrismCart(x[i], y[j], z, mass)
                Vxx1[i, j], Vxy1[i, j], Vxz1[i, j], Vyy1[i, j] = V[:4]
                Vyz1[i, j], Vzz1[i, j], P1[i, j], Vz1[i, j] = V[4:]

            if object in ['point', 'all']:
                V = mpoinCart(x[i], y[j], z, mass)
                Vxx2[i, j], Vxy2[i, j], Vxz2[i, j], Vyy2[i, j] = V[:4]
                Vyz2[i, j], Vzz2[i, j], P2[i, j], Vz2[i, j] = V[4:]
        update_progress(j/float(len(x)))
    sys.stdout.write("\n")

    if plotting:
        if object in ['prisma', 'all']:
            title = 'Prism-Mass Model | height = %f m' % z
            plot_gravity(Vxx1, Vxy1, Vxz1, Vyy1, Vyz1, Vzz1, P1, Vz1, title)
        if object in ['point', 'all']:
            title = 'Point-Mass Model | height = %f m' % z
            plot_gravity(Vxx2, Vxy2, Vxz2, Vyy2, Vyz2, Vzz2, P2, Vz2, title)
    if object == 'prisma':
        return Vxx1, Vxy1, Vxz1, Vyy1, Vyz1, Vzz1, P1, Vz1
    elif object == 'point':
        return Vxx2, Vxy2, Vxz2, Vyy2, Vyz2, Vzz2, P2, Vz2


def hollow_sphere(a=3, b=6):
    r, g = mHollowSphere(a, b, N=250)
    plot_hollow_sphere(a, b, r, g)
    return
