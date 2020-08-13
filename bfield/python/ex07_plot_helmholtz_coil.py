################################################################################
#
#  BFIELD
#
#   Simple example of plot of the magnitude of the magnetic field
#   produced by a Helmholtz coil
#
#
################################################################################

import numpy as np
import bfield
import matplotlib.pyplot as plt

# Loops ( Ra,I0,Nturns, Xcenter,Ycenter,Zcenter, Ux,Uy,Uz )
Loops = np.array([[ 0.200,100,10,  0.200,0,0, 1,0,0 ],
                  [ 0.200,100,10, -0.200,0,0, 1,0,0 ] ])
Nloops = np.size(Loops,0)

X = np.linspace( -0.4, 0.4, 100 )
Y = np.linspace( -0.4, 0.4, 100 )
Bnorm = np.zeros((X.size,Y.size))

for i in range(0,X.size):
  for j in range(0,Y.size):
    for k in range(0,Nloops):
      Ra     = Loops[k][0]
      I0     = Loops[k][1]
      Nturns = Loops[k][2]
      Center = Loops[k][3:6]
      Uhat   = Loops[k][6:9]
      Point = np.array([ X[i], Y[j], 1e-10 ])
      Bx,By,Bz = bfield.loopxyz( Ra,I0,Nturns,Center,Uhat, Point )
      Bnorm[i][j] += np.sqrt( Bx*Bx + By*By + Bz*Bz )

plt.figure(1)
XX,YY = np.meshgrid(X,Y)
plt.contourf(np.transpose(XX),np.transpose(YY),Bnorm,30)
plt.colorbar()
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('B-field magnitude [T] - Helmholtz Coil')
plt.savefig('ex07_plot_helmholtz_coil_bnorm.png',dpi=150)
plt.show()

plt.figure(2)
plt.plot(X,Bnorm[:,0]*1e4)
plt.xlabel('Axis [m]')
plt.ylabel('B [Gauss]')
plt.title('B-field magnitude along axis [T] - Helmholtz Coil')
plt.savefig('ex07_plot_helmholtz_coil_Baxis.png',dpi=150)
plt.show()
