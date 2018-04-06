import numpy as np
from Gravity.functions import mPrismCart, mpoinCart
import matplotlib.pyplot as plt

"""
v 1.0
 Gravity modelling
   with point mass and prism models

 Author: I. Vasconcelos  2016
 Translated to python: S. Schneider 2018
"""


def model_gravity(object='all'):
    """
    object: 'prisma', 'point' or 'all'
    """

    for scal in [1, 10]:
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
            # g = waitbar(j/length(x))
    """
    h=figure;
    set(h,'Name',['Prism-Mass Model | height =' num2str(z,2) 'm'])
    subplot 321
            imagesc(x,y,P1(:,:))
            axis equal
            colorbar
            title('Potential [m^2/s^2]')
    subplot 322
            imagesc(x,y,Vz1(:,:))
            axis equal
            view(2)
            colorbar
            title('Gravity anomaly [m/s^2]')
    subplot 323
            imagesc(x,y,Vxx1(:,:))
            axis equal
            shading flat
            view(2)
            colorbar
            title('Gravity gradient Vxx [1/s^2]')
    subplot 324
            imagesc(x,y,Vyy1(:,:))
            axis equal
            shading flat
            view(2)
            colorbar
            title('Gravity gradient Vyy [1/s^2]')
    subplot 325
            imagesc(x,y,Vzz1(:,:))
            axis equal
            shading flat
            view(2)
            colorbar
            title('Gravity gradient Vzz [1/s^2]')
    subplot 326
            imagesc(x,y,Vxz1(:,:))
            axis equal
            shading flat
            view(2)
            colorbar
            title('Gravity gradient Vxz [1/s^2]')
            set(gcf,'Color','white')

    h=figure;
    set(h,'Name',['Point-Mass Model | height =' num2str(z,2) 'm'])
    subplot 321
            imagesc(x,y,P2(:,:))
            axis equal
            colorbar
            title('Potential [m^2/s^2]')
    subplot 322
            imagesc(x,y,Vz2(:,:))
            axis equal
            view(2)
            colorbar
            title('Gravity anomaly [m/s^2]')
    subplot 323
            imagesc(x,y,Vxx2(:,:))
            axis equal
            shading flat
            view(2)
            colorbar
            title('Gravity gradient Vxx [1/s^2]')
    subplot 324
            imagesc(x,y,Vyy2(:,:))
            axis equal
            shading flat
            view(2)
            colorbar
            title('Gravity gradient Vyy [1/s^2]')
    subplot 325
            imagesc(x,y,Vzz2(:,:))
            axis equal
            shading flat
            view(2)
            colorbar
            title('Gravity gradient Vzz [1/s^2]')
    subplot 326
            imagesc(x,y,Vxz2(:,:))
            axis equal
            shading flat
            view(2)
            colorbar
            title('Gravity gradient Vxz [1/s^2]')
            set(gcf,'Color','white')
    """
    return
