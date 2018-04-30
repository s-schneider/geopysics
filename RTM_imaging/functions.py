from RTM_imaging.data import Marmousi, migration
import numpy as np
import matplotlib.pyplot as plt
import time


# Data source

migration_path = migration.__path__[0]
marmousi_path = Marmousi.__path__[0]


def load_model(type):
    if type == 'marmousi':
        """
        EXAMPLE 2 : Marmousi model
        """
        VelocityModel = np.genfromtxt('%s/marmhard.dat' % marmousi_path)
        VelocityModel = VelocityModel.reshape([122, 384])

        VelocityModel0 = np.genfromtxt('%s/marmsmooth.dat' % marmousi_path)
        VelocityModel0 = VelocityModel.reshape([122, 384])
        # Write new variable
        velocityModel = VelocityModel[21:121, 241:341]
        velocityModel0 = VelocityModel0[21:121, 241:341]

    else:
        """
        EXAMPLE 1 : Layered model
        """
        velocityModel0 = 3000. * np.ones([100, 100])
        velocityModel = velocityModel0.copy()
        velocityModel[50:52] = 4000

        return velocityModel, velocityModel0


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


def set_FD_params(Vp, V0, dx=24, dz=24):
    Vp = Vp.transpose()
    V0 = V0.transpose()

    nz, nx = Vp.shape[:]

    Mdz = np.ones(Vp.shape) * dz
    dt = 0.2 * (Mdz/Vp/np.sqrt(2)).min()

    vmin = Vp.min()

    # determine time samples nt from wave travelime to depth and back to
    # surface
    nt = int(np.round(np.sqrt((dx*nx)**2 + (dz*nx)**2) * (2/vmin/dt) + 1))
    t = np.arange(0, nt)*dt

    # add region around model for applying absorbing boundary conditions (20
    # nodes wide)
    Vm = np.vstack((
                   np.matlib.repmat(Vp[0], 20, 1),
                   Vp,
                   np.matlib.repmat(Vp[-1], 20, 1)
                   ))

    Vm = np.hstack((
                   Vm,
                   np.matlib.repmat(Vm[:, -1], 20, 1).transpose()
                   ))

    Vm0 = np.vstack((
                   np.matlib.repmat(V0[0], 20, 1),
                   V0,
                   np.matlib.repmat(V0[-1], 20, 1)
                   ))

    Vm0 = np.hstack((
                   Vm0,
                   np.matlib.repmat(Vm0[:, -1], 20, 1).transpose()
                   ))

    Vm = Vm.transpose()
    Vm0 = Vm0.transpose()

    return Vm, Vm0, t, dt, nt


def ricker(f=None, n=None, dt=None, t0=None, t1=None):
    """
    RICKER creates an causal ricker wavelet signal

       RICKER creates and plots a default causal ricker wavelet with:

           peak frequency   = 20 Hz
           sampling time    = 0.001 seconds
           number of points = 100;
           peak location    = 1/F = 1/20Hz

       RW = RICKER(...) returns the default wavelet in RW.

       [RW,T] = RICKER(...) returns the time vector in T.

       Specifying parameters of peak frequency (F, Hz), number of points (N),
       and sampling time (DT) are specified by the syntax:

           [RW,T] = RICKER(F)
           [RW,T] = RICKER(F,N)
           [RW,T] = RICKER(F,N,DT)

       [RW,T] = RICKER(F,N,DT,T0) creates a ricker wavelet with peak centered
       at T0.

       [RW,T] = RICKER(F,N,DT,T0,T1) creates a 2 dimensional symmetric
       ricker wavelet with sift in 1st dimension of T0 and second dimension of
       T1.

       Example 1:
           ricker % plots a 20 Hz Ricker Wavelet over 0.1 seconds

       Example 2:
        % create a ricker wavelet with 40 Hz, 200 points, and 0.02 s between
        % samples
        [rw,t] = ricker(40,200,0.002);
        plot(t,rw), xlabel('Time'), ylabel('Amplitude')
    """
    # Define inputs if needed

    if f and n and dt and t0:
        if t1 is not None:
            is2d = True
        else:
            is2d = False
    elif f and n and dt:
        t0 = 1/float(f)
        is2d = False
    elif f and n:
        dt = 0.001
        t0 = 1/float(f)
        is2d = False
    elif f:
        n = 100
        dt = 0.001
        t0 = 1/float(f)
        is2d = False
    else:
        f = 20.
        n = 100
        dt = 0.001
        t0 = 1/f
        is2d = False

    # Create the wavelet and shift in time if needed
    T = dt*(n-1)
    t = np.arange(0, T+dt, dt)
    tau = t-t0
    if not is2d:
        rw = (
             (1-tau * tau * f**2. * np.pi**2.) *
             np.exp(-tau**2. * np.pi**2. * f**2.)
            )
    else:
        t1, t2 = np.meshgrid(tau, t-t1)
        rw = (
             (1-(t1**2. + t2**2.) * f**2. * np.pi**2.) *
             np.exp(-(t1**2. + t2**2.) * np.pi**2. * f**2.)
            )

    return rw, t


def generate_shots(Vp, Vm, Vm0, dt, nt, dx=24, dz=24, animation=True):
    f = 60.
    nz, nx = Vp.shape[:]
    x = np.arange(1, nx+1) * dx
    z = np.arange(1, nz+1) * dz

    data = np.zeros((nt, nx))

    if animation:
        fig = plt.figure()
        ax = range(4)
        cbar = range(3)
        clim = [-1000, 1000]

        ax[0] = plt.subplot2grid((5, 5), (0, 0), colspan=2, rowspan=2)
        ax[1] = plt.subplot2grid((5, 5), (0, 3), colspan=2, rowspan=2)
        ax[2] = plt.subplot2grid((5, 5), (3, 0), colspan=2, rowspan=2)
        ax[3] = plt.subplot2grid((5, 5), (3, 3), colspan=2, rowspan=2)

        # subplot(2,2,1)
        iVp = ax[0].imshow(Vp, extent=(dx, nx*dx, nz*dz, dz), cmap='seismic')
        ax[0].plot(x[0], z[0], 'x')
        ax[0].plot(x[0], z[0], '^', color='white')
        ax[0].set_title('c(x)')
        ax[0].set_xlabel('Distance (m)')
        ax[0].set_xlim(dx, nx*dx)
        ax[0].set_ylabel('Depth (m)')
        ax[0].set_ylim(nz*dz, 0)
        cbar[0] = fig.colorbar(iVp, ax=ax[0])

        nxi = 1
        for ixs in range(21, 22+nxi):  # shot loop
            # initial wavefield
            rw, t = ricker(f, nz+40, dt, dt*ixs, 0)
            rw = rw[0:nz+20]

            # generate shot records
            tic = time.time()
            [data, snapshot] = fm2d(Vm, rw, dz, dx, nt, dt)
            toc = time.time()
            msg = "Elapsed time is %s seconds." % (toc-tic)
            print(msg)

            tic = time.time()
            [data0, snapshot0] = fm2d(Vm0, rw, dz, dx, nt, dt)
            toc = time.time()
            msg = "Elapsed time is %s seconds." % (toc-tic)
            print(msg)

            data = data[21:-20, :].transpose()
            data0 = data0[21:-20, :].transpose()
            dataS = data - data0

            #  save(['Marmousi/snapshot0',num2str(ixs-20),'.mat'],'snapshot0');
            #  save(['Marmousi/shotfdm',num2str(ixs-20),'.mat'],'data')
            #  save(['Marmousi/shotfdmS',num2str(ixs-20),'.mat'],'dataS')

            # plot initial wavefield
            """
            set(hshot,'XData',x(ixs-20),'YData',z(1));
            subplot(2,2,2)
            imagesc(x,z,rw(1:end-20,21:end-20))
            xlabel('Distance (m)'); ylabel('Depth (m)');
            title(['Shot ',num2str(ixs-20),' at ',num2str(x(ixs-20)),' m']);
            colormap(seismic(1024))

            if ismember(ixs-20,[1 nx/2 nx])
                start = 1;
            else
                start = nt;
            end

            for i = start:10:nt
                % plot shot record evolution
                ds = zeros(nt,nx);
                ds(1:i,:) = data(1:i,:);
                subplot(2,2,3)
                imagesc(x,t,ds)
                xlabel('Distance (m)'), ylabel('Time (s)')
                title('Shot Record')
                %caxis([-0.5 0.5]) % this for layered model
                caxis([-5 5]) % this for Marmousi model

                % plot wave propagation
                subplot(2,2,4)
                imagesc(x,z,snapshot(1:end-20,21:end-20,i))
                xlabel('Distance (m)'), ylabel('Depth (m)')
                title(['Wave Propagation t = ',num2str(t(i),'%10.3f')])
                %caxis([-5 5]) % this for layered model
                caxis([-50 50]) % this for Marmousi model


                drawnow;
            end
            end %shot loop
            """
    return


def fm2d(v, model, dz, dx, nt, dt):
    """
    DEBUGGING IN PROGRESS!

    model(nz,nx)      model vector
    v(nz,nx)          velocity model
    nx                number of horizontal samples
    nz                number of depth samples
    nt                numer of time samples
    dx                horizontal distance per sample
    dz                depth distance per sample
    dt                time difference per sample
    """
    # add grid points for boundary condition
    #  np.matlib.repmat(Vm0[:, -1], 20, 1)
    # model = [repmat(model(:,1),1,20), model, repmat(model(:,end),1,20)];
    # model(end+20,:) = model(end,:);

    # v = [repmat(v(:,1),1,20), v, repmat(v(:,end),1,20)];
    # v(end+20,:) = v(end,:);

    # Initialize storage
    nz, nx = model.shape
    data = np.zeros((nx, nt))
    fdm = np.zeros((nz, nx, 3))

    # Boundary Absorbing Model
    iz = np.arange(20)
    boundary = (np.exp(-((0.005 * (20-iz))**2)))**10.
    boundary = boundary.transpose()

    # Forward-Time Modeling
    fdm[:, :, 1] = model
    data[:, 0] = model[0, :]

    # finite difference coefficients
    a = (v*dt/dx)**2    # wave equation coefficient
    b = 2-4*a

    # common indicies
    izs = 1                      # interior z
    ize = nz-1
    ixs = 1                    # interior x
    ixe = nx-1

    izb = np.arange(0, nz-20)      # boundary z

    snapshot = np.zeros((nz, nx, nt))

    for it in np.arange(2, nt):
        # finite differencing on interior
        fdm[izs:ize, ixs:ixe, 2] = (
                          b[izs:ize, ixs:ixe] * fdm[izs:ize, ixs:ixe, 1] -
                          fdm[izs:ize, ixs:ixe, 0] +
                          a[izs:ize, ixs:ixe] * (fdm[izs:ize, ixs+1:ixe+1, 1] +
                                                 fdm[izs:ize, ixs-1:ixe-1, 1] +
                                                 fdm[izs+1:ize+1, ixs:ixe, 1] +
                                                 fdm[izs-1:ize-1, ixs:ixe, 1])
                          )

        # finite differencing at ix = 1 and ix = nx (surface, bottom)
        fdm[izs:ize, 0, 2] = (
                         b[izs:ize, 0] * fdm[izs:ize, 0, 1] -
                         fdm[izs:ize, 0, 0] +
                         a[izs:ize, 0] * (fdm[izs:ize, 1, 1] +
                                          fdm[izs+1:ize+1, 0, 1] +
                                          fdm[izs-1:ize-1, 0, 1])
                        )

        fdm[izs:ize, nx-1, 2] = (
                          b[izs:ize, nx-1] * fdm[izs:ize, nx-1, 1] -
                          fdm[izs:ize, nx-1, 0] +
                          a[izs:ize, nx-1] * (fdm[izs:ize, nx-2, 1] +
                                              fdm[izs+1:ize+1, nx-1, 1] +
                                              fdm[izs-1:ize-1, nx-1, 1])
                         )

        # finite differencing at iz = 1 and iz = nz (z boundaries)
        fdm[0, ixs:ixe, 2] = (
                         b[0, ixs:ixe] * fdm[0, ixs:ixe, 1] -
                         fdm[0, ixs:ixe, 0] +
                         a[0, ixs:ixe] * (fdm[1, ixs:ixe, 1] +
                                          fdm[0, ixs+1:ixe+1, 1] +
                                          fdm[0, ixs-1:ixe-1, 1])
                        )

        fdm[nz-1, ixs:ixe, 2] = (
                          b[nz-1, ixs:ixe] * fdm[nz-1, ixs:ixe, 1] -
                          fdm[nz-1, ixs:ixe, 0] +
                          a[nz-1, ixs:ixe] * (fdm[nz-2, ixs:ixe, 1] +
                                              fdm[nz-1, ixs+1:ixe+1, 1] +
                                              fdm[nz-1, ixs-1:ixe-1, 1])
                         )

        # finite differencing at four corners (1,1), (nz,1), (1,nx), (nz,nx)
        fdm[0, 0, 2] = (
                        b[0, 0] * fdm[0, 0, 1] - fdm[0, 0, 0] +
                        a[0, 0] * (fdm[1, 0, 1] + fdm[0, 1, 1])
                       )
        fdm[nz-1, 0, 2] = (
                         b[nz-1, 0] * fdm[nz-1, 0, 1] - fdm[nz-1, 0, 0] +
                         a[nz-1, 1] * (fdm[nz-1, 1, 1] + fdm[nz-2, 0, 1])
                        )
        fdm[0, nx-1, 2] = (
                         b[0, nx-1] * fdm[0, nx-1, 1] - fdm[0, nx-1, 0] +
                         a[0, nx-1] * (fdm[0, nx-2, 1] + fdm[2, nx-1, 1])
                        )
        fdm[nz-1, nx-1, 2] = (
                          b[nz-1, nx-1] * fdm[nz-1, nx-1, 0] -
                          fdm[nz-1, nx-1, 0] +
                          a[nz-1, nx-1] * (fdm[nz-2, nx-1, 1] +
                                           fdm[nz-1, nx-2, 1])
                         )

        # update fdm for next time iteration
        fdm[:, :, 0] = fdm[:, :, 1]
        fdm[:, :, 1] = fdm[:, :, 2]

        # apply absorbing boundary conditions to 3 sides (not surface)
        for ixb in range(20):
            fdm[izb, ixb, 0] = boundary[ixb] * fdm[izb, ixb, 0]
            fdm[izb, ixb, 1] = boundary[ixb] * fdm[izb, ixb, 1]
            ixb2 = nx-20+ixb
            fdm[izb, ixb2, 0] = boundary[nx-ixb2-1] * fdm[izb, ixb2, 0]
            fdm[izb, ixb2, 1] = boundary[nx-ixb2-1] * fdm[izb, ixb2, 1]
            izb2 = nz-20+ixb
            fdm[izb2, :, 0] = boundary[nz-izb2-1] * fdm[izb2, :, 0]
            fdm[izb2, :, 1] = boundary[nz-izb2-1] * fdm[izb2, :, 1]

        # update data
        data[:, it] = fdm[0, :, 1]

        snapshot[:, :, it] = fdm[:, :, 1]

    data = data[21:nx-20, :]

    return data, snapshot
