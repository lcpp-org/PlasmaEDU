clear all
close all
clc

global CurrentLoops

% CurrentLoops = [  Xc,  Yc,  Zc,  nx,  ny,  nz,  I0,   Ra,   N_windings; ...
  CurrentLoops = [ -0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 10e3, 0.025, 1; ...
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10e3, 0.05,  1; ... 
                    0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 10e3, 0.10,  1; ...
                    0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 10e3, 0.05,  1; ...
                    0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 10e3, 0.025, 1]; 


% Open a figure
figureNumber = 1;
figure( figureNumber )

% 3D plot of current loops
plot3D_currentloops( CurrentLoops, 100, figureNumber )

% Grid of points for contour map of B-field isolines 
Px =  -0.1 : 0.2/45  : 0.30;
Py = 0.001 : 0.15/40 : 0.15;
Pz = 0.0;

% Contour map of B-field isolines
Bnorm = zeros(size(Px,2),size(Py,2));
for i=1:size(Px,2)
    for j=1:size(Py,2)
        P = [ Px(i); Py(j); Pz ];
        B = bfield_currentloops( P, CurrentLoops );
        Bnorm(i,j) = sqrt( B(1)*B(1) + B(2)*B(2) + B(3)*B(3) ); 
    end
end
[XX,YY] = meshgrid(Px,Py);
contour(XX',YY',Bnorm, 250)
axis equal

% Fieldlines
% (Set: coordinates of the seed points, direction, length along 's', ds_maxstep  )
Y0x       = [ -0.1*ones(1,11),  0.1*ones(1,11),  0.1*ones(1,11),  0.3*ones(1,11)  ];
Y0y       = [  0.0:0.003:0.03,  0.0:0.006:0.06,  0.0:0.006:0.06,  0.0:0.003:0.03  ];
Y0z       = [  0.0*ones(1,11),  0.0*ones(1,11),  0.0*ones(1,11),  0.0*ones(1,11)  ];
direction = [  1.0*ones(1,11),  1.0*ones(1,11), -1.0*ones(1,11), -1.0*ones(1,11)  ];
length    = [  0.4*ones(1,11),  0.3*ones(1,11),  0.3*ones(1,11),  0.4*ones(1,11)  ];
options   = odeset( 'maxstep', 2.0e-3 );
for i=1:numel(Y0y)
    Y0 = [ Y0x(i); Y0y(i); Y0z(i); direction(i) ];
    [s,y] = ode113( @ blines, [ 0.0 length(i) ], Y0, options);
    plot3( y(:,1), y(:,2), y(:,3), 'k-' )
end

% Set a view 
view([0 0 1])

% Print figure on file 
print( '-f1', '-dpdf', 'fig01' )
