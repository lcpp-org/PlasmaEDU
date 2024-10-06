#  Simple 1D Diffusion-Advection-Reaction problem
#
#   du          du         d^2 u
#  ----  +  Vx ----  =  D -------  +  Src(x)
#   dt          dx         d x^2
#
#  Crank Nicholson Scheme on the diffusion part

import numpy as np
import matplotlib.pyplot as plt
import math
import copy

# MATERIAL PROPERTIES

# Tungsten
#Cp     =  1340            # Specific heat [J/kg K]
#rho    =  19300           # Density [kg/m^3]
#kappa  =  173             # Thermal conductivity [W·m-1·K-1]
#D      =  kappa/rho/Cp    # Thermal diffusivity  [m^2/s]

# Lithium
Cp     =  4270.0	#3570	# Specific heat [J/kg K]
rho    =   505.0	#535	# Density [kg/m^3]
kappa  =  45.979	#84.8	# Thermal conductivity [W·m-1·K-1]
D      =  kappa/rho/Cp		# Thermal diffusivity  [m^2/s]

# Choose the Boundary Type on left side
# BoundaryFlag = 1  : Dirichlet
# BoundaryFlag = 2  : Neumann

BoundaryFlag = 2

T_amb = 0.0
Twall = 0.0
dT_dr = 0.0
if(BoundaryFlag == 1):
	T_amb = 300
	Twall = 800
elif(BoundaryFlag == 2):
	T_amb  = 300
	q_flux = 10.0e6 # Heat flux on the left wall [W/m^2]
	dT_dr  = - q_flux/kappa

# Space grid [m]
x1   = 0.000
x2   = 0.002
nptx = 50
h    = (x2-x1)/(nptx-1)
x    = np.linspace(x1, x2, nptx)

# Set Courant number
alphaC = -0.5

# Advection velocity [m/s]
v = 0.00

# Time span [s]
t1   = 0.0
t2   = 1.0
if (v!=0.0):
	# satisfying Courant condition
	dt = abs(alphaC*h/v)
else:
	dt = (t2-t1)/500
	alphaC = 0.0

nptt = math.floor((t2-t1)/dt)+1
time = np.linspace(t1, t2, nptt)

# Build matrix of the implicit problem
A = np.zeros((nptx,nptx))

# Source vector
Src = 0.0*np.ones(nptx)

# matrix values
beta = D*dt/(h**2)

lm  = -alphaC/4 - beta/2
lcc =         1 + beta
lcp =  alphaC/4 - beta/2

cm  =  beta/2 + alphaC/4
cc  =       1 - beta
cp  =  beta/2 - alphaC/4

# matrix boundary left
if(BoundaryFlag == 1):
	A[0,0] =  1
elif(BoundaryFlag == 2):
	A[0,0] = -3
	A[0,1] =  4
	A[0,2] = -1

# matrix inside
for i in range(1,nptx-1):
	A[i,i-1] = lm
	A[i,i]   = lcc
	A[i,i+1] = lcp

# matrix boundary right
A[nptx-1,nptx-1] = 1

# Initial condition
y = np.zeros(nptx)
for i in range(0,nptx):
	#y[i] = 1 + 2*exp(-(x(i)-10)^2)
	y[i] = T_amb
yo = copy.deepcopy(y)

vecr, T_surf = np.zeros(nptx), np.zeros(nptt)
for n in range(0,nptt):
	t = n*dt
	if(BoundaryFlag==1):
		vecr[0]    = Twall
		vecr[nptx-1,0] = T_amb
	elif(BoundaryFlag==2):
		vecr[0]    = dT_dr*2*h
		vecr[nptx-1] = T_amb

	for i in range(1,nptx-1):
		vecr[i] = cm*yo[i-1] + cc*yo[i] + cp*yo[i+1] + Src[i]

	y = np.linalg.solve(A,vecr)
	yo = copy.deepcopy(y)

	if (n%10)==0:
		plt.plot(x*1000, y-273.15, 'm', linewidth = 1.0)
	# Surface temperature
	T_surf[n] = y[0]

# 	if (n%20)==0:
		# Run plasma model
		# os.system('./zapdos -i input_file.i')

		# Get solution from plasma model
		# q_flux = np.genfromtxt('read_total_heat_flux.csv')
		# dT_dr = q_flux/kappa

plt.xlabel('x [mm]')
plt.ylabel('T [$^\circ$C]')
plt.grid(True)
plt.title('Temperature profile evolution')
plt.show()
