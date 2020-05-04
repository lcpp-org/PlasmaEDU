%  Simple 1D Diffusion-Advection-Reaction problem
%
%   du          du         d^2 u
%  ----  +  Vx ----  =  D -------  +  Src(x)
%   dt          dx         d x^2
%
%  Crank Nicholson Scheme on the diffusion part

clear all
close all
clc

% MATERIAL PROPERTIES

% Tungsten
Cp     =  1340;            % Specific heat [J/kg K]
rho    =  19300;           % Density [kg/m^3]
kappa  =  173;             % Thermal conductivity [W·m-1·K-1]
D      =  kappa/rho/Cp;    % Thermal diffusivity  [m^2/s]

% Lithium
% Cp     =  3570;            % Specific heat [J/kg K]
% rho    =  535;             % Density [kg/m^3]
% kappa  =  84.8;            % Thermal conductivity [W·m-1·K-1]
% D      =  kappa/rho/Cp;    % Thermal diffusivity  [m^2/s]

% Choose the Boundary Type on left side
% BoundaryFlag = 1  : Dirichlet
% BoundaryFlag = 2  : Neumann

BoundaryFlag = 2;

switch BoundaryFlag
    case 1
        T_amb = 300;
        Twall = 800;
    case 2
        T_amb  = 300;
        q_flux = 10.0e6; % Heat flux on the left wall [W/m^2]
        dT_dr = - q_flux / kappa;
end

% Space grid [m]
x1   = 0.000;
x2   = 0.020;
nptx = 20;
h    = (x2-x1)/(nptx-1);
x    = linspace( x1, x2, nptx );

% Set Courant number
alphaC = -0.5;

% Advection velocity [m/s]
v = 0.00;

% Time span [s]
t1   = 0.0;
t2   = 1.0;
if (v~=0.0)
    % dt satisfying Courant condition
    dt = abs(alphaC*h/v);
else
    dt = (t2-t1)/500;
    alphaC = 0.0;
end
nptt = floor((t2-t1)/dt)+1;
time = linspace( t1, t2, nptt );


% Build matrix of the implicit problem
A = zeros(nptx);

% Source vector
Src = 0.0*ones(nptx);

% matrix values
beta = D*dt/(h^2);

lm = -alphaC/4 - beta/2;
lcc = 1 + beta;
lcp = alphaC/4 - beta/2;

cm = beta/2 + alphaC/4;
cc = 1 - beta;
cp = beta/2 - alphaC/4;

% matrix boundary left
switch BoundaryFlag
   case 1
      A(1,1) =  1;
   case 2
      A(1,1) = -3;
      A(1,2) =  4;
      A(1,3) = -1;
end

% matrix inside
for i=2:(nptx-1)
    A(i,i-1) = lm;
    A(i,i)   = lcc;
    A(i,i+1) = lcp;
end

% matrix boundary right
A(nptx,nptx) = 1;

% Initial condition
for i=1:(nptx)
    %y(i) = 1 + 2*exp(-(x(i)-10)^2);
    y(i) = T_amb;
end
yo = y;

plot(x,yo,'m','LineWidth',2.0)
% axis([min(x) max(x) 0 5])
xlabel('x')
ylabel('y')
grid on
hold on

tic
for n=1:nptt
    t = n*dt;
    switch BoundaryFlag
        case 1
            vecr(1,1)    = Twall;
            vecr(nptx,1) = T_amb;
        case 2
            vecr(1,1)    = dT_dr*2*h;
            vecr(nptx,1) = T_amb;
    end
    for i=2:(nptx-1)
        vecr(i,1) = cm*yo(i-1) + cc*yo(i) + cp*yo(i+1) + Src(i);
    end
    y = A\vecr;
    yo = y;
    if mod(n,2)==0
        plot( x*1000, y-273.15, 'b' )
    end
    % Surface temperature
    T_surf(n) = y(1);
end
toc
xlabel('x [mm]')
ylabel('T [C]')
title('Temperature profile evolution')
