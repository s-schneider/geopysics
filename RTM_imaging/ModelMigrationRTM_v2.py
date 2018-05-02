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
travelTime = generate_traveltimes(Vp0, plot=True, save=True)


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

reverse_time_migration(Vm, Vm0, dV, dataS, shots, t, dt, nt)

"""
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
