from RTM_imaging.data import Marmousi, migration
from RTM_imaging.plotting import (_init_shot_plot, plot_initial_wavefield,
                                  plot_wavefield_animation,
                                  plot_travel_times)
import numpy as np
import scipy.io
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

    return rw


def generate_shots(Vp, Vm, Vm0, t, dt, nt, dx=24, dz=24, animation=True,
                   shots=1):
    f = 60.
    nz, nx = Vp.shape[:]
    x = np.arange(1, nx+1) * dx
    z = np.arange(1, nz+1) * dz

    data = np.zeros((nt, nx))

    ax, fig, hshot = _init_shot_plot(Vp, dx, dz, nx, nz, x, z)

    for ixs in range(21, 21+shots):  # shot loop
        # initial wavefield
        rw = ricker(f, nz+40, dt, dt*ixs, 0)
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

        data = data.transpose()  # [21:-20, :]
        data0 = data0.transpose()  # [21:-20, :]

        # plot initial wavefield
        ax = plot_initial_wavefield(hshot, ax, dx, dz, nx, nz, ixs-20,
                                    x[ixs-20],  rw[:-20, 21:-20])

        plt.pause(0.01)
#            colormap(seismic(1024))

        if ixs-20 in [1, nx/2, nx]:
            start = 0
        else:
            start = nt

        if animation:
            plot_wavefield_animation(ax, fig, start, nt, 10,
                                     nt, nx, nz, dx, dz,
                                     data, snapshot, t)
        else:
            plot_wavefield_animation(ax, fig, nt-1, nt, 1,
                                     nt, nx, nz, dx, dz,
                                     data, snapshot, t)
            plt.pause(0.01)

    return data, data0


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

    data = data[21:nx-19, :]

    return data, snapshot


def generate_traveltimes(Vp0, dx=24, dz=24, plot=True):

    nz, nx = Vp0.shape[:]
    x = np.arange(1, nx+1) * dx

    travelTime = np.zeros((nz, nx, nx))

    if plot:
        fig, ax, hshot, im_tT = plot_travel_times(Vp0, None, None, dx, dz,
                                                  init=True)

    for ixs in range(nx):
        travelTime[:, :, ixs] = ray2d(Vp0, [0, ixs], dz)
        if plot:
            fig, ax, hshot, im_tT = plot_travel_times(Vp0, x[ixs], ixs, dx, dz,
                                                      travelTime[:, :, ixs],
                                                      ax=ax, fig=fig,
                                                      im_tT=im_tT,
                                                      hshot=hshot)
    return travelTime


def ray2d(V, Shot, dx):
    """
     2D ray-tracing
       RAY2D
    """
    # load dA
    dA = scipy.io.loadmat('%s/dA.mat' % migration_path)['dA']
    dA= dA.reshape(169, 144)
    # Constants
    v0 = 10000.
    sz = 6
    sx = 6
    sz2 = 2*sz + 1
    sx2 = 2*sx + 1
    zs = Shot[0] + sz
    xs = Shot[1] + sx

    # Derived values
    V = 1./V           # convert to slowness
    mxV = V.max()
    nz, nx = V.shape

    # Preallocate
    T = np.ones((nz+sz2, nx+sx2)) * v0
    M = T.copy()
    S = np.ones((nz+sz2-1, nx+sx2-1))

    #
    M[sz:nz+sz+1, sx:nx+sx+1] = 0

    S[sz:nz+sz, sx:nx+sx] = V
    S[nz+sz, sx:nx+sx] = 2*S[nz+sz-1, sx:nx+sx] - S[nz+sz-2, sx:nx+sx]
    S[sz:nz+sz, nx+sx] = 2*S[sz:nz+sz, nx+sx-1] - S[sz:nz+sz, nx+sx-1]
    S[nz+sz, nx+sx] = 2*S[nz+sz, nx+sx] - S[nz+sz-1, nx+sx-1]

    T[zs, xs] = 0
    M[zs, xs] = v0

    AS = S[-sz+zs:sz+zs, -sx+xs:sx+xs]
    TT = T[-sz+zs:sz+zs+1, -sx+xs:sx+xs+1]

    dAAS = np.dot(dA, AS.flatten()) + T[zs, xs]
    dAAS = dAAS.reshape(sz2, sx2)
    T[-sz+zs:sz+zs+1, -sx+xs:sx+xs+1] = np.minimum(dAAS, TT)

    mxT = T[zs-1:zs+2, xs-1:xs+2].max()

    while True:
        indx = T+M <= mxT+mxV

        if not indx.any():
            indx = M == 0

        idz, idx = np.where(indx is True)
        M[indx] = v0

        for z, x in zip(idz, idx):
            mxT = np.maximum(mxT, T[x, z])
            AS = S[-sz+z:sz+z, -sx+x:sx+x]
            TT = T[-sz+z:sz+z+1, -sx+x:sx+x+1]
            dAAS = np.dot(dA, AS.flatten()) + T[z, x]
            dAAS = dAAS.reshape(sz2, sx2)
            T[-sz+z:sz+z+1, -sx+x:sx+x+1] = np.minimum(dAAS, TT)

        if np.all(M[sz:nz+sz, sx:nx+sx]):
            break

        mxT = T[idz, idx].max()

    T = T[sz:nz+sz, sx:nx+sx]*dx
    return T
