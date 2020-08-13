################################################################################
#
#  BFIELD
#
#   Simple example of plot of the magnitude of the magnetic field
#   produced by a current loop, solving Biot-Savart
#
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import bfield

# Simple Current Loop, discretized in Npoints
Ra       = 0.05
Center   = np.array([0, 0, 0])
Uhat     = np.array([0, 1, 0])
Npoints  = 100
filament = bfield.makeloop( Ra, Center, Uhat, Npoints )

current  = 1000
X = np.linspace(  0.0,   0.1, 20 )
Y = np.linspace( -0.05, 0.05, 20 )
Z = 0.0
Bnorm = np.zeros((X.size,Y.size))
point = np.zeros((3,1))

for i in range(0,X.size):
  for j in range(0,Y.size):
    point[0] = X[i]
    point[1] = Y[j]
    point[2] = Z
    Bx, By, Bz = bfield.biotsavart( filament, current, point )
    Bnorm[i][j] = np.sqrt(Bx*Bx + By*By + Bz*Bz)

plt.figure(1)
XX,YY = np.meshgrid(X,Y)
plt.contourf(np.transpose(XX),np.transpose(YY),Bnorm,30)
plt.colorbar()
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.title('B-field magnitude [T] of a Current Loop')
plt.savefig('ex10_plot_filament_simple_loop.png',dpi=150)
plt.show()
