# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_velocity_model(Vp, Vp0, dx=24, dz=24):

    nz, nx = Vp.shape[:]
    dV = Vp - Vp0

    x = np.arange(1, nx+1) * dx
    z = np.arange(1, nz+1) * dz

    fig = plt.figure()
    ax = range(3)
    cbar = range(3)
    clim = [-1000, 1000]

    ax[0] = plt.subplot2grid((18, 18), (0, 0), colspan=6, rowspan=6)
    iVp = ax[0].imshow(Vp, extent=(dx, nx*dx, nz*dz, dz), cmap='seismic')
    ax[0].plot(x[0], z[0], '^', color='white')  # , mew=10, ms=15)
    ax[0].set_title('c(x)')
    ax[0].set_xlabel('Distance (m)')
    ax[0].set_xlim(dx, nx*dx)
    ax[0].set_ylim(nz*dz, 0)
    ax[0].set_ylabel('Depth (m)')
    cbar[0] = fig.colorbar(iVp)

    ax[1] = plt.subplot2grid((18, 18), (0, 12), colspan=6, rowspan=6)
    iVp0 = ax[1].imshow(Vp0, extent=(dx, nx*dx, nz*dz, dz), cmap='seismic')
    ax[1].set_title(r'$c_{0}(x)$')
    ax[1].set_xlabel('Distance (m)')
    ax[1].set_ylabel('Depth (m)')
    cbar[1] = fig.colorbar(iVp0)

    ax[2] = plt.subplot2grid((18, 18), (10, 0), colspan=6, rowspan=6)
    idV = ax[2].imshow(dV, extent=(dx, nx*dx, nz*dz, dz), cmap='seismic',
                       clim=clim)
    ax[2].set_title(r'${\delta}c(x)$')
    ax[2].set_xlabel('Distance (m)')
    ax[2].set_ylabel('Depth (m)')
    cbar[2] = fig.colorbar(idV)

    return


def _init_shot_plot(Vp, dx, dz, nx, nz, x, z):
    plt.ion()
    fig = plt.figure()
    ax = range(4)
    cbar = range(3)

    ax[0] = plt.subplot2grid((5, 5), (0, 0), colspan=2, rowspan=2)
    ax[1] = plt.subplot2grid((5, 5), (0, 3), colspan=2, rowspan=2)
    ax[2] = plt.subplot2grid((5, 5), (3, 0), colspan=2, rowspan=2)
    ax[3] = plt.subplot2grid((5, 5), (3, 3), colspan=2, rowspan=2)

    # subplot(2,2,1)
    iVp = ax[0].imshow(Vp, extent=(dx, nx*dx, nz*dz, dz), cmap='seismic')
    hshot = ax[0].plot(x[0], z[0], '*', color='white')[0]
    ax[0].set_title('c(x)')
    ax[0].set_xlabel('Distance (m)')
    ax[0].set_xlim(dx, nx*dx)
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_ylim(nz*dz, 0)
    cbar[0] = fig.colorbar(iVp, ax=ax[0])

    return ax, fig, hshot


def plot_initial_wavefield(hshot, ax, dx, dz, nx, nz, x_no, x, rw):
    hshot.set_xdata(x)
    ax[1].imshow(rw, extent=(dx, nx*dx, nz*dz, dz),
                 cmap='seismic')
    ax[1].set_xlabel('Distance (m)')
    ax[1].set_ylabel('Depth (m)')
    ax[1].set_title('Shot %s at %s m' % (x_no, x))
    return ax


def plot_wavefield_animation(ax, fig, start, end, step, nt, nx, nz, dx, dz,
                             data, snapshot, t):

    for i in np.arange(start, end, step):
        # plot shot record
        ds = np.zeros((nt, nx))
        ds[0:i, :] = data[0:i, :]
        if i == start:
            im1 = ax[2].imshow(ds, extent=(dx, nx*dx, t.max(), 0),
                               aspect='auto')
            im2 = ax[3].imshow(snapshot[0:-20, 21:-19, i],
                               extent=(dx, nx*dx, nz*dz, dz))
            ax[2].set_xlabel('Distance (m)')
            ax[2].set_ylabel('Time (s)')
            ax[2].set_title('Shot Record')
            # %caxis([-0.5 0.5]) % this for layered model
            # caxis([-5 5]) % this for Marmousi model

            ax[3].set_xlabel('Distance (m)')
            ax[3].set_ylabel('Depth (m)')
            ax[3].set_title('Wave Propagation t = %.3f' % t[i])
            # %caxis([-5 5]) % this for layered model
            # caxis([-50 50]) % this for Marmousi model
        else:

            im1.set_data(ds)
            im2.set_data(snapshot[0:-20, 21:-19, i])
            ax[3].set_title('Wave Propagation t = %.3f' % t[i])
            im1.autoscale()
            im2.autoscale()
            fig.canvas.draw()

        plt.pause(1E-16)


def plot_scattered_wave_data(Vp, Vp0, data, data0, t, dx=24, dz=24,
                             model_name=None):
    
    if model_name == 'marmousi':
        vmin = -1
        vmax = 1
    else:
        vmin = -0.15
        vmax = 0.15

    fig = plt.figure()
    nz, nx = Vp.shape[:]
    dV = Vp - Vp0
    dataS = data - data0

    gs = gridspec.GridSpec(5, 5)
    ax = range(4)

    ax[0] = plt.subplot(gs[0:2, 0:2])
    ax[0].set_xlabel('Distance (m)')
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_title(r'$\delta V$')
    im0 = ax[0].imshow(dV, extent=(dx, nx*dx, nz*dz, dz),
                       cmap='gray', vmin=-1000, vmax=1000)
    fig.colorbar(im0, ax=ax[0])

    ax[1] = plt.subplot(gs[0:2, 3:5])
    im1 = ax[1].imshow(dataS, extent=(dx, nx*dx, t.max(), 0), aspect='auto',
                       cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_xlabel('Distance (m)')
    ax[1].set_ylabel('Time (s)')
    ax[1].set_title(r'$d_S = d - d_0$')
    fig.colorbar(im1, ax=ax[1])

    ax[2] = plt.subplot(gs[3:5, 0:2])
    im2 = ax[2].imshow(data, extent=(dx, nx*dx, t.max(), 0), aspect='auto',
                       cmap='gray', vmin=vmin, vmax=vmax)
    ax[2].set_xlabel('Distance (m)')
    ax[2].set_ylabel('Time (s)')
    ax[2].set_title(r'$d$')
    fig.colorbar(im2, ax=ax[2])


    ax[3] = plt.subplot(gs[3:5, 3:5])
    im3 = ax[3].imshow(data0, extent=(dx, nx*dx, t.max(), 0), aspect='auto',
                       cmap='gray', vmin=vmin, vmax=vmax)
    ax[3].set_xlabel('Distance (m)')
    ax[3].set_ylabel('Time (s)')
    ax[3].set_title(r'$d_0$')
    fig.colorbar(im3, ax=ax[3])



def plot_travel_times(Vp0, shot_x, ixs, dx, dz, travelTime=None, init=False,
                      ax=None, fig=None, im_tT=None, hshot=None):
    nz, nx = Vp0.shape[:]
    x = np.arange(1, nx+1) * dx
    z = np.arange(1, nz+1) * dz
    if init:
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2)
        ax = range(2)

        ax[0] = plt.subplot(gs[0, 0])
        ax[0].imshow(Vp0, extent=(dx, nx*dx, nz*dz, dz), cmap='seismic',
                     vmin=-1000, vmax=1000)
        ax[0].set_xlabel('Distance (m)')
        ax[0].set_ylabel('Depth (m)')
        ax[0].set_title(r'$c_0(x)$')
        hshot = ax[0].plot(x[0], z[0], '*', color='white')[0]
    
        ax[1] = plt.subplot(gs[0, 1])
        ax[1].set_xlabel('Distance (m)')
        ax[1].set_ylabel('Depth (m)')
    
    else:
        if ixs == 0:
            im_tT = ax[1].imshow(travelTime, extent=(dx, nx*dx, nz*dz, dz),
                                 cmap='seismic', vmin=0, vmax=1)
#            fig.colorbar(im_tT, ax=ax[1])
            ax[1].set_title('Traveltime for shot %i' % ixs)
        else:
            ax[1].set_title('Traveltime for shot %i' % ixs)
            im_tT.set_data(travelTime)
            im_tT.autoscale()
            im_tT.set_clim(0, 1)
            fig.canvas.draw()
            hshot.set_xdata(shot_x)
    plt.pause(0.0005)
   

    return fig, ax, hshot, im_tT
