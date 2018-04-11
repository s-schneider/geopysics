function m = ShotKirchPSDM_v2(travelTime,shot,dt,dz,nz,ixs,dx,nx,aper2depth,twin)
% Migrate a shot record for a given travel time between shot (source) and
% gather (receiver) using a simple Kirchoff Migration algorithm
%
% Inputs:
%   travelTime      travel time array
%   shot            shot array (nz,nx)
%   dt              sampling time
%   nz,nx              number of samples in z,x directions
%   ixs             shot location in shot
%   aper2depth      maximum aperturture-to-depth ratio
%   twin            time window to sum over
%
% Outputs:
%   m               migrated image (nz,nx)

nt = size(shot,1);
nr = size(shot,2);
if nr ~= nx 
    error('number of receivers must be equal to nx!'); 
end
m = zeros(nz,nx);

itwin = round((twin/2)/dt);

tic

for iz = 1:nz % loop over depth points
    for ix = 1:nx
        % get traveltime to shot
        soutt = travelTime(iz,ix,ixs);
        
        % maximum aperture
        apmax = aper2depth * ((iz-1)*dz);
        
        % loop over receivers
        for ixr = 1:nr
            if (ixr-1)*dx < apmax % if offset smaller than max aperture
           % get traveltime to receiver
            rectt = travelTime(iz,ix,ixr);
            
           % total traveltime, rounded for time index
           it = round( (soutt + rectt) / dt ) +1;
           %ixs
           %ixr 
           %it
           % image contribution
           for itt = 1:2*itwin
               if ((it-itwin)+(itt-1)) > 0 &&  ((it-itwin)+(itt-1)) < nt
               m(iz,ix) = m(iz,ix) + shot((it-itwin)+(itt-1),ixr)/(2*itwin);
               end
           end
            end
        end
        
        
    end
end

%parfor ixr = 1:nx
% for ixr = 1:nx
%     it = shot2RecTime(travelTime,ixs,ixr,dt,nx);
%     m = m + reshape( shot(it,ixr), nz, nx);
% end
toc