import math
import numpy as np

# Physical Constants - SI Units (unused for test cases)
# qe   = 1.60217646e-19
# me   = 9.10938188e-31
# mp   = 1.67262158e-27
# KB   = 1.38000000e-23
# mu0  = 4*np.pi*1e-7
# lux  = 299792458
# eps0 = 1.0/(mu0*lux*lux)

# Using normalized units
eps0 = 1.0

# Species 1 - Electrons
Q1   = -1.0
M1   =  1.0
T1   =  1.0
Vth1 =  np.sqrt(T1)
Vb   =  0.0            # Drift velocity (0.0 for single uniform Maxwellian)

n1   = 1.0
Te   =  T1
LD   = np.sqrt(T1)
source_freq = 5.e-4

# Time [s]
Nt   = 1000
dt   = 0.02
time_vector = np.linspace(0.0, Nt*dt, Nt)



###################################
###  Physical Space Parameters  ###
###################################
Nx   = 128              # Number of gridpoints in X
Nv   = 128              # Number of gridpoints in V
k    = 0.5              # Wavenumber
Xmax = np.pi/k
L    = 2.0*np.pi / k
X    = np.linspace( 0, L, Nx )
dX   = abs(X[2] - X[1])



###################################
###        Velocity grid        ###
###################################
Vmin = Vth1
Vmax = Vth1
V    = np.linspace( -6*Vmax, 6*Vmax, Nv )
dV   = abs(V[2] - V[1])
beta = 0.5*dt/dV

# CFL warning booleans
cfl_space_warning = 0
cfl_velocity_warning = 0


###################################
###       Boundary Values       ###
###################################
# Ghost points are included in each dimension - value depends on the stencil
# size of the selected numerical scheme.
# RECOMMENDED: Keep at default value of 3, which accounts for the largest stencil
# currently implemented. (Fourth-order Upwind method)
# I1 and I2 are the left- and right-most physical space indices
# J1 and J2 are the upper- and lower-most physical velocity indices
gp = 3

I1 = gp
I2 = Nx+(gp-1)
J1 = gp
# J2 = 66
J2 = Nv+(gp-1)
NXDIM = Nx+(2*gp)
NVDIM = Nv+(2*gp)


###################################
###  Meshgrid (for plotting)    ###
###################################
[XX,VV] = np.meshgrid(X, V)
XX = XX.transpose()
VV = VV.transpose() / Vmax

#######################################################################
###                     Plotting Options                            ###
###                                                                 ###
###  plot_on ......... turn on plotting routine
###                    VALUE: 0 = off   ;   1 = on
###
###  plot_save ....... save figure(s) to output file
###                    VALUE: 0 = off   ;   1 = on
###                    Figures are plotted in output file, /OUT/.
###                    Directory is automatically made if it does not exist.
###
###
###  plot_show ....... display plots while simulation is running
###                    VALUE: 0 = off   ;   1 = on
###
###  nplot ........... integer indicating when plotting routine
###                    should run. Plots will be made every
###                    [nplot] timesteps, including t = 0.
###
###
###
######################################################################
plot_on   = 1
plot_save = 1
plot_show = 1
nplot     = 25

###################################
###  Initial Distribution       ###
###################################
f1  = np.zeros((NXDIM,NVDIM))
perturb = 0.10                    # Perturbation of initial density
nx = 1.0 - perturb*np.cos(k*X)
for i in range(I1,I2+1):
    f1[i,J1:J2+1] = nx[i-I1]/np.sqrt( 2.0*np.pi*Vth1**2.0) * np.exp(-0.5*(V-Vb)**2.0/(Vth1**2.0) )


######################################
###      Numerical Scheme          ###
###  1: First-order Upwind         ###
###  2: Second-order Upwind        ###
###  3: Third-order Upwind         ###
###  4: Fourth-order Upwind        ###
###  Note: add CFL condition check ###
######################################

numerical_scheme = 2
