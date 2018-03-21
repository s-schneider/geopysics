from RTM_imaging.data import Marmousi, migration
import numpy as np
import matplotlib.pyplot as plt

"""
Seismic Migration Example - Layered & Marmousi Models

Matlab Codes by Ivan Vascensoles, 2018
Python version by Simon Schneider, 2018
"""
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

    fig, ax = plt.subplots()
    ax = range(3)
    cbar = range(3)
    clim = [-1000, 1000]

    ax[0] = plt.subplot2grid((18, 18), (0, 0), colspan=6, rowspan=6)
    iVp = ax[0].imshow(Vp, extent=(dx, nx*dx, nz*dz, dz), cmap='seismic')
    ax[0].plot(x[0], z[0], 'x')
    ax[0].set_title('c(x)')
    ax[0].set_xlabel('Distance (m)')
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


"""
PART 1 :
Read in velocity model data and plot it
"""
# load velocityModel

Vp, Vp0 = load_model('migration')

dx=24
dz=24

plot_velocity_model(Vp, Vp0, dx, dz)

"""
PART 2 :

Set FD modelling parameters
Use the velocity model to simulate a seismic survey.  The wave equation
is solved using finite differences for a defined initial wavefield.

calculate time step dt from stability crierion for finite difference
solution of the wave equation.
"""

dt = 0.2 * min(min(dz/Vp/np.sqrt(2)))

"""
% determine time samples nt from wave travelime to depth and back to
% surface
vmin = min(velocityModel(:));
nt = round(sqrt((dx*nx)^2 + (dz*nx)^2)*2/vmin/dt + 1);
t  = (0:nt-1).*dt;

% add region around model for applying absorbing boundary conditions (20
% nodes wide)
V = [repmat(velocityModel(:,1),1,20) velocityModel repmat(velocityModel(:,end),1,20)];
V(end+1:end+20,:) = repmat(V(end,:),20,1);
V0 = [repmat(velocityModel0(:,1),1,20) velocityModel0 repmat(velocityModel0(:,end),1,20)];
V0(end+1:end+20,:) = repmat(V0(end,:),20,1);

% Define frequency parameter for ricker wavelet
f  = 60;
%f  = 40;

%%
%%%%
%%%% PART 3 :
%%%%
%% Generate shots and save to file


data = zeros(size(nt,nx));


figure(gcf)

figure
subplot(2,2,1)
imagesc(x,z,velocityModel)
xlabel('Distance (m)'); ylabel('Depth (m)');
title('Velocity Model');
hold on
hshot = plot(x(1),z(1),'w*');
hold off
colormap(gray)
colormap(seismic(1024))

%nxi = nx;
nxi = 1;
for ixs = 21:21+nxi % shot loop
    % initial wavefield
    rw = ricker(f,nz+40,dt,dt*ixs,0);
    rw = rw(1:nz+20,:);

    % generate shot records
    ixs
    tic
    [data, snapshot] = fm2d(V,rw,nz,dz,nx,dx,nt,dt);
    toc
    tic
    [data0, snapshot0] = fm2d(V0,rw,nz,dz,nx,dx,nt,dt);
    toc

    data = data(21:end-20,:)';
    data0 = data0(21:end-20,:)';
    dataS = data - data0;

%     save(['Marmousi/snapshot0',num2str(ixs-20),'.mat'],'snapshot0');
%     save(['Marmousi/shotfdm',num2str(ixs-20),'.mat'],'data')
%     save(['Marmousi/shotfdmS',num2str(ixs-20),'.mat'],'dataS')

%     % plot initial wavefield
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


%%
%%%%
%%%% PART 4 :
%%%%
%% Plotting scattered-wave data

figure
subplot(2,2,1)
imagesc(x,z,dV)
xlabel('Distance (m)'); ylabel('Depth (m)');
title('\deltaV');
caxis([-1000 1000])
hold on
hshot = plot(x(1),z(1),'w*');
hold off

subplot(2,2,2)
imagesc(x,t,dataS)
xlabel('Distance (m)'), ylabel('Time (s)')
title('d_S = d - d_0')
%caxis([-0.15 0.15])
caxis([-0.5 0.5]) % this for layered model
caxis([-1 1]) % this for marmousi

subplot(2,2,3)
imagesc(x,t,data)
xlabel('Distance (m)'), ylabel('Time (s)')
title('d')
%caxis([-0.15 0.15])
caxis([-0.5 0.5]) % this for layered model
caxis([-1 1]) % this for marmousi

subplot(2,2,4)
imagesc(x,t,data0)
xlabel('Distance (m)'), ylabel('Time (s)')
title('d_0')
%caxis([-0.15 0.15])
caxis([-0.5 0.5]) % this for layered model
caxis([-1 1]) % this for marmousi

colormap(gray(1024))
%colormap(seismic(1024))

%%
%%%%
%%%% PART 5 :
%%%%
%% Traveltime by 2D ray-tracing
% Generate the traveltime field for all z = 0 locations
%vidObj = VideoWriter('FaultModelTravelTime.avi');
%open(vidObj);
travelTime = zeros(nz,nx,nx);

figure
subplot(2,2,1)
imagesc(x,z,velocityModel0)
xlabel('Distance (m)'); ylabel('Depth (m)');
title('c_0(x)');
hold on
hshot = plot(x(1),z(1),'w*');
hold off
colormap(gray)
colormap(seismic(1024))


subplot(2,2,2)
for ixs = 1:nx
    travelTime(:,:,ixs) = ray2d(Vp0,[1 ixs],dx);
    imagesc(x,z,travelTime(:,:,ixs))
    xlabel('Distance (m)'), ylabel('Depth (m)')
    title(['Traveltime for shot ',num2str(ixs)])
    caxis([0. 1.])
    colorbar
    set(hshot,'XData',x(ixs));
    drawnow
    %writeVideo(vidObj,getframe(gcf));
end
%close(vidObj)
%save results for later re-use
save('Marmousi/travelTime.mat', 'travelTime')

%%
%%%%
%%%% PART 6 :
%%%%
%% Process Shots - Kirchhoff Migration

%vidObj = VideoWriter('videos\FaultModelKirchhoff.avi');
%open(vidObj);
%load('travelTime.mat');
Stacked = zeros(nz,nx);
figure(gcf)
subplot(2,2,1)
imagesc(x,z,dV)
xlabel('Distance (m)'); ylabel('Depth (m)');
title('{\delta}c(x)');
caxis([-1000 1000])
hold on
hshot = plot(x(1),z(1),'w*');
hold off

colormap  seismic %bone
colormap  gray %bone
nxii = nxi;
%nxii = nx;
Stacked=0;
MM = zeros(nz,nx,nxii);
for ixs = 1:nxii
    %load(['shotfdmS',num2str(ixs),'.mat'])
    %shot = dataS(21:end-20,:);
    shot = dataS(:,:);
    M = ShotKirchPSDM_v2(travelTime,shot,dt,dz,nz,ixs,dx,nx,8.0,0.02);
    MM(:,:,ixs) = M;
    Stacked = sum(MM,3)/nxii;

    subplot(2,2,2)
    imagesc(x,z,Stacked)
    xlabel('Distance (m)'); ylabel('Depth (m)');
    title('Stacked Image');
    caxis([-20 20])

    subplot(2,2,3)
    imagesc(x,t,shot)
    xlabel('Distance (m)'); ylabel('Time (s)');
    title(['Current Shot ',num2str(ixs)]);
    caxis([-0.3 0.3])

    subplot(2,2,4)
    imagesc(x,t,M)
    xlabel('Distance (m)'); ylabel('Time (s)');
    title(['Current Migrated Shot ',num2str(ixs)]);
    caxis([-20 20])

    set(hshot,'XData',x(ixs));

    drawnow
    %writeVideo(vidObj,getframe(gcf));
end
%close(vidObj);

%%
%%%%
%%%% PART 7 :
%%%%
%% Process Shots - Reverse Time Migration

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
