from RTM_imaging.functions import (generate_shots, load_model, set_FD_params,
                                   generate_traveltimes)
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

dx = 24
dz = 24

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
data, data0 = generate_shots(Vp, Vm, Vm0, t, dt, nt, shots=1, animation=False)


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
travelTime = generate_traveltimes(Vp0)


"""
PART 6 :
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
