function m = KirchhoffPSDM_v2(travelTime,shot,dt,nz,ixs,nx)
% Migrate a shot record for a given travel time between shot (source) and
% gather (receiver) using a simple Kirchoff Migration algorithm
%
% Inputs:
%   travelTime      travel time array
%   shot            shot array (nz,nx)
%   dt              sampling time
%   nz,nx              number of samples in z,x directions
%   ixs             shot location in shot
%
% Outputs:
%   m               migrated image (nz,nx)


nx = size(shot,2);
m = zeros(nz,nx);

tic

for iz = 1:nz % loop over depth points
    for ix = 1:nx
        
    end
end

%parfor ixr = 1:nx
% for ixr = 1:nx
%     it = shot2RecTime(travelTime,ixs,ixr,dt,nx);
%     m = m + reshape( shot(it,ixr), nz, nx);
% end
toc