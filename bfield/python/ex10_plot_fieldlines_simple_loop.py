################################################################################
#
#  BFIELD
#
#   Simple example of plot of the fieldlines
#   produced by a current loop
#
#
################################################################################

import bfield
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Simple Current Loop, discretized in Npoints
Ra       = 0.05
Center   = np.array([0, 0, 0])
Angles   = np.array([0, 0, 0]) * np.pi/180.0
Npoints  = 100
filament = bfield.makeloop( Ra, Center, Angles, Npoints )

current  = 1000
X = np.linspace(  0.0, 0.1, 30 )
Y = np.linspace( -0.05, 0.05, 30 )
Z = 0.0
Bnorm = np.zeros((X.size,Y.size))
point = np.zeros((3,1))

# Plot B-field first
for i in range(0,X.size):
  for j in range(0,Y.size):
    point[0] = X[i]
    point[1] = Y[j]
    point[2] = Z
    Bx, By, Bz = bfield.biotsavart(filament, current, point)
    Bnorm[i][j] = np.sqrt(Bx*Bx + By*By + Bz*Bz)

plt.figure(1)
XX,YY = np.meshgrid(X,Y)
plt.contourf(np.transpose(XX),np.transpose(YY),Bnorm,30)
plt.colorbar()

# Now solve Field Lines

# Initial position of the field line
Nlines = 10
fieldlines_X0     = np.linspace( 0, Ra*0.98, Nlines )
fieldlines_Y0     = np.linspace( 0,   0,  Nlines )
fieldlines_Z0     =  np.linspace( 0,   0,  Nlines )
fieldlines_direction = np.ones( Nlines )
fieldlines_length = np.ones( Nlines ) * 0.05

for i in range(np.size(fieldlines_X0,0)):
    # Top portion
    Y0 = np.array([ fieldlines_X0[i], fieldlines_Y0[i],fieldlines_Z0[i],fieldlines_direction[i]])
    interval = x=np.arange(0.0,fieldlines_length[i],1e-4)
    fieldlines = odeint(bfield.blines, Y0, interval, args=(filament, current) )
    print fieldlines
    plt.plot( fieldlines[:,0], fieldlines[:,1], 'r-' )
    # Bottom portion
    Y0 = np.array([ fieldlines_X0[i], fieldlines_Y0[i],fieldlines_Z0[i],-fieldlines_direction[i]])
    interval = x=np.arange(0.0,fieldlines_length[i],1e-4)
    fieldlines = odeint(bfield.blines, Y0, interval, args=(filament, current) )
    print fieldlines
    plt.plot( fieldlines[:,0], fieldlines[:,1], 'r-' )

plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.title('B-field magnitude [T] and Fieldlines of a Current Loop')
plt.savefig('ex09_plot_fieldlines_simple_loop.png',dpi=150)

plt.show()
