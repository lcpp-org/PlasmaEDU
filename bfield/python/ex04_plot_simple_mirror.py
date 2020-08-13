################################################################################
#
#  BFIELD
#
#   Simple example of plot of the magnitude of the magnetic field
#   produced by two current loops (simple Magentic Mirror)
#
#
################################################################################

import numpy as np
import bfield
import matplotlib.pyplot as plt

# Loops ( Ra,I0,Nturns, Xcenter,Ycenter,Zcenter, Ux,Uy,Uz )
Loops = np.array([[ 0.02,100,1,  0.04,0,0, 1,0,0 ],
                  [ 0.02,100,1, -0.04,0,0, 1,0,0 ] ])
Nloops = np.size(Loops,0)

X = np.linspace( -0.1, 0.1, 100 )
Y = np.linspace( -0.1, 0.1, 100 )
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
plt.title('B-field magnitude [T] - Simple Magnetic Mirror')
plt.savefig('ex04_plot_simple_mirror.png',dpi=150)
plt.show()
