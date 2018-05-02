from RTM_imaging.functions import (generate_shots, load_model, set_FD_params,
                                   generate_traveltimes,
                                   kirchhof_migration)
from RTM_imaging.plotting import (plot_velocity_model,
                                  plot_scattered_wave_data)


"""
Seismic Migration Example - Layered & Marmousi Models

Matlab Codes by Ivan Vascensoles, 2018
Python version by Simon Schneider, 2018
"""

"""
PART 1 :
Read in velocity model data and plot it
"""
# load velocityModel

model_name = 'migration'

Vp, Vp0 = load_model(model_name)
dV = Vp - Vp0
dx = 24
dz = 24
shots = 1
plot_velocity_model(Vp, Vp0, dx, dz)

"""
PART 2 :

Set FD modelling parameters
Use the velocity model to simulate a seismic survey.  The wave equation
is solved using finite differences for a defined initial wavefield.

calculate time step dt from stability crierion for finite difference
solution of the wave equation.
"""

Vm, Vm0, t, dt, nt = set_FD_params(Vp, Vp0)

# Define frequency parameter for ricker wavelet


"""
PART 3 :

Generate shots and save to file
"""
data, data0 = generate_shots(Vp, Vm, Vm0, t, dt, nt, shots=shots,
                             animation=False, save=True)


"""
PART 4 :

Plotting scattered-wave data
"""
plot_scattered_wave_data(Vp, Vp0, data, data0, t, model_name=model_name)


"""
PART 5 :

Traveltime by 2D ray-tracing
Generate the traveltime field for all z = 0 locations
"""
travelTime = generate_traveltimes(Vp0, plot=False, save=True)


"""
PART 6 :
Process Shots - Kirchhoff Migration
"""
dataS = data - data0
Stacked = kirchhof_migration(Vp, dV, dataS, shots, t, dt)


"""
PART 7 :
Process Shots - Reverse Time Migration
"""

Stacked = reverse_time_migration(Vp, Vp0, dataS, shots, t, dt, nt)


def reverse_time_migration(Vp, Vp0, dataS, n_of_shots, t, dt, nt, tmax='all',
                           plot=True, loadfile=True, travelTime=None,
                           dx=24, dz=24):
    """
    tmax: 'all' or float between 0-1
    """

    if loadfile:
        travelTime = load_file('travelTime.dat', (100, 100, 100))
    else:
        if travelTime is None:
            raise IOError('No travelTime data given')

    nz, nx = Vp.shape

    for ixs in range(n_of_shots):
        if loadfile:
            shape = (2668, 100)
            dataS = load_file('shotfdmS%s.dat' % str(ixs+1), shape)
            shape = (120, 140, 2668)
            snapshot0 = load_file('snapshot0%s.dat' % str(ixs+1), shape)

        shot = np.zeros((Vp.shape[1], nt))
        shot[21:-20, :] = dataS.transpose().copy()
        ntmig = shot.shape[1]

        print('Migrating shot %s/%s ' % (str(ixs+1), n_of_shots))
        rtmsnapshot = rtmod2d(Vp0, shot, nz, dz, nx, dx, ntmig, dt)

        M = np.zeros(snapshot0.shape[:2])
        s2 = np.zeros(snapshot0.shape[:2])
        if tmax == 'all':
            t = t[-1]
        for i in np.arange(nt, step=10):
            if t[nt-i] < tmax:
                M += snapshot0[:, :, nt-i] * rtmsnapshot[:, :, nt-i]
                s2 += np.power(snapshot0[:, :, i], 2)

        if plot:
            Mdiff = np.diff(M[:-19, 21:-19], 2, 0)
            ss0 = snapshot0(1:end-20,21:end-20,nt-i+1)
            rmt = rtmsnapshot(1:end-20,21:end-20,nt-i+1)
            if ixs == 0:
                hshot, fig, ax = plot_rtmigration(dV, ss0, rtm,
                                                  Mdiff, shot, 0, t, nt, dx,
                                                  dz, init=True)
            else:
                hshot, fig, ax = plot_rtmigration(dV, ss0, rtm,
                                                  Mdiff, shot, 0, t, nt, dx,
                                                  dz, init=True)

    return Stacked

def plot_rtmigration(dV, S snapshot0, rtmsnapshot, Mdiff, shot, i, t, nt, 
                     dx, dz, hshot=None, init=False,  fig=None, ax=None):

    nz, nx = dV.shape
    x = np.arange(1, nx+1) * dx
    z = np.arange(1, nz+1) * dz

    if fig is None:
        fig = plt.figure()
        gs = gridspec.GridSpec(5, 5)
        ax = range(4)

    ax[0] = plt.subplot(gs[0:2, 0:2])
    ax[0].imshow(dV, extent=(dx, nx*dx, nz*dz, dz), cmap='gray',
                 vmin=-1000, vmax=1000)
    ax[0].set_xlabel('Distance (m)')
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_title(r'$\delta c(x)$')
    if init:
        hshot = ax[0].plot(x[0], z[0], '*', color='white')[0]
    else:
        hshot.set_xdata(x[i])

    ax[1] = plt.subplot(gs[0:2, 3:5])
    im1 = ax[1].imshow(snapshot0, extent=(dx, nx*dx, nz*dz, dz),
                       aspect='auto', cmap='gray', vmin=-20, vmax=20)
    ax[1].set_xlabel('Distance (m)')
    ax[1].set_ylabel('Depth (m)')
    ax[1].set_title(r'Forward Time Wave Propagation t = %.3f' % t(nt-i+1))
    fig.colorbar(im1, ax=ax[1])

    ax[2] = plt.subplot(gs[3:5, 0:2])
    im2 = ax[2].imshow(rtmsnapshot, extent=(dx, nx*dx, nz*dz, dz),
                       aspect='auto', cmap='gray', vmin=-20, vmax=20)
    ax[2].set_xlabel('Distance (m)')
    ax[2].set_ylabel('Depth (m)')
    ax[2].set_title(r'Reverse Time Wave Propagation')
    fig.colorbar(im2, ax=ax[2])

    ax[3] = plt.subplot(gs[3:5, 3:5])
    im3 = ax[3].imshow(Mdiff, extent=(dx, nx*dx, nz*dz, dz), aspect='auto',
                       cmap='gray', vmin=-30, vmax=30)
    ax[3].set_xlabel('Distance (m)')
    ax[3].set_ylabel('Depth (m)')
    ax[3].set_title(r'Current Migrated Shot %s' % i)
    fig.colorbar(im3, ax=ax[3])

    return hshot, fig, ax

"""

%vidObj = VideoWriter('videos\FaultModelRTM.avi');
%open(vidObj);
Stacked = zeros(nz+20,nx+40);
%colormap seismic %bone

figure
subplot(2,2,1)
imagesc(x,z,dV)
xlabel('Distance (m)'); ylabel('Depth (m)');
title('{\delta}c(x)');
caxis([-1000 1000])
hold on
hshot = plot(x(1),z(1),'r*','MarkerSize',10);
hold off
colormap(gray(1024))
%colormap(seismic)

nxiii = nxi;
%nxiii = nx;

for ixs = 1:nxiii
    %load(['shotfdmS',num2str(ixs),'.mat'])
    shot = zeros(size(V,2),nt);
    shot(21:end-20,:) = dataS(:,:)';
    ntmig = size(shot,2);

    tic
    [~, rtmsnapshot] = rtmod2d(V0,shot,nz,dz,nx,dx,ntmig,dt);
    toc
    %save(['Marmousi/rtmsnapshot',num2str(ixs),'.mat'],'rtmsnapshot');

    %load(['snapshot',num2str(ixs),'.mat']);

    M = 0;
    s2 = 0;
    %tmax = t(nt);  % use all time samples
    tmax = 0.9; % why use only a portion of time samples?
    for i = 1:10:nt
        %M = snapshot0(:,:,i).*rtmsnapshot(:,:,nt-i+1)+M;
        if t(nt-i+1) < tmax
        M = snapshot0(:,:,nt-i+1).*rtmsnapshot(:,:,nt-i+1)+M;
        s2 = snapshot0(:,:,i).^2+s2;
        end

        if ismember(ixs,[1 nx/2 nx])
            %figure
            subplot(2,2,3)
            imagesc(x,z,snapshot0(1:end-20,21:end-20,nt-i+1))
            xlabel('Distance (m)'); ylabel('Depth (m)');
            title(['Forward Time Wave Propagation t = ',num2str(t(nt-i+1),'%10.3f')])
            caxis([-1 1])
            caxis([-10 10]) % this for layered
            %caxis([-10 10]) % this for marmousi

            subplot(2,2,4)
            %imagesc(x,z,rtmsnapshot(1:end-20,21:end-20,nt-i+1))
            imagesc(x,z,rtmsnapshot(1:end-20,21:end-20,nt-i+1))
            xlabel('Distance (m)'); ylabel('Depth (m)');
            title('Reverse Time Wave Propagation')
            caxis([-1 1])
            caxis([-100 100]) % this for layered
            caxis([-300 300]) % this for marmousi

            subplot(2,2,2)
            %imagesc(x,z,diff(M(1:end-20,21:end-20)./s2(1:end-20,21:end-20),2,1))
            imagesc(x,z,diff(M(1:end-20,21:end-20),2,1))
            xlabel('Distance (m)'); ylabel('Depth (m)');
            title(['Current Migrated Shot ',num2str(ixs)]);
            %caxis([-10 10]) % set this for all time samples
            caxis([-30 30]) % set this for tmax = 0.9s
            caxis([-2000 2000]) % this for layered
            caxis([-8000 8000]) % this for marmousi

            drawnow
            %writeVideo(vidObj,getframe(gcf));
        end
    end

end
%close(vidObj);

%%
%%%%
%%%% PART 8 : Marmousi model only
%%%%
%% RTM - Full survey

IStacked = zeros(nz,nx);
figure(gcf)
subplot(2,2,1)
imagesc(x,z,dV)
xlabel('Distance (m)'); ylabel('Depth (m)');
title('{\delta}c(x)');
caxis([-1000 1000])
hold on
hshot = plot(x(1),z(1),'w*');
hold off
colormap(gray(1024))

nxiv = nxi;
%nxiv = nx;
Stacked=0;
II = zeros(nz,nx,nxiv);
for ixs = 1:nx
    tic
    load(['shotfdmS',num2str(ixs),'.mat'])
    load(['snapshot0',num2str(ixs),'.mat'])
    load(['rtmsnapshot',num2str(ixs),'.mat'])
    shot = dataS(:,:);
    souw = snapshot0(1:100,21:120,:);
    recw = rtmsnapshot(1:100,21:120,:);
    I = sum(( souw .* recw),3);
    II(:,:,ixs) = I;
    IStacked = sum(II,3)/nxiv;
    toc

    subplot(2,2,2)
    imagesc(x,z,diff(IStacked,2,1))
    xlabel('Distance (m)'); ylabel('Depth (m)');
    title('Stacked RTM');
    caxis([-20 20])
    caxis([-100000 10000]) % this for marmousi

    subplot(2,2,3)
    imagesc(x,t,shot)
    xlabel('Distance (m)'); ylabel('Time (s)');
    title(['Current Shot ',num2str(ixs)]);
    caxis([-0.3 0.3])

    subplot(2,2,4)
    imagesc(x,t,diff(I,2,1))
    xlabel('Distance (m)'); ylabel('Time (s)');
    title(['Current RTM Shot ',num2str(ixs)]);
    caxis([-20 20])
    caxis([-90000 90000]) % this for marmousi

    set(hshot,'XData',x(ixs));

    drawnow
    %writeVideo(vidObj,getframe(gcf));
end
%close(vidObj);
"""
