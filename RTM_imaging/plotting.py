# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np



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