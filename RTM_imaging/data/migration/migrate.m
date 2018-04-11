function m = migrate(travelTime,shot,dt,nz,ixs,nx)
% Migrate a shot record for a given travel time between shot (source) and
% gather (receiver) using a simple Kirchoff Migration algorithm
%
% Inputs:
%   travelTime      travel time array
%   shot            shot array (nz,nx)
%   dt              sampling time
%   nz              number of samples in z direction
%   ixs             shot location in shot
%
% Outputs:
%   m               migrated image (nz,nx)

% Copyright 2010 The MathWorks, Inc.
% All rights reserved

nx = size(shot,2);
m = zeros(nz,nx);

tic
parfor ixr = 1:nx
    it = shot2RecTime(travelTime,ixs,ixr,dt,nx);
    m = m + reshape( shot(it,ixr), nz, nx);
end
toc