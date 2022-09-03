################################################################################
#
#  BFIELD
#
#   Simple example of a toroidal magnetic field obtained from a combination
#   of 40 Loops (HIDRA case)
#
#
################################################################################

import bfield
import numpy as np
import matplotlib.pyplot as plt

# HIDRA geometry
R0 = 0.5        # [m] Major Radius
a0 = 0.19        # [m] Minor Radius
Ra = a0 + 0.1265 # [m] Avg Coil Radius
Ncoils = 40      # [m] Number of toroidal coils
I0 = 770         # [A] Current
Nturns = 13      # [#] Number of turns on each coil

# Construct the input for the 40 coils
phi = np.linspace(0.0, 2*np.pi, Ncoils)
Loops = np.zeros((Ncoils,9))
for i in range(0,Ncoils):
  Xcenter = R0 * np.cos( phi[i] )
  Ycenter = R0 * np.sin( phi[i] )
  Zcenter = 0.0
  Angle1  = np.pi - phi[i]
  Angle2  = 0.0
  Angle3  = 0.0
  Loops[i,0] = Ra
  Loops[i,1] = I0
  Loops[i,2] = Nturns
  Loops[i,3] = Xcenter
  Loops[i,4] = Ycenter
  Loops[i,5] = Zcenter
  Loops[i,6] = Angle1
  Loops[i,7] = Angle2
  Loops[i,8] = Angle3

# Points of interest along the midplane
X = np.linspace( -1, 1, 50 )
Y = np.linspace( -1, 1, 50 )
Bnorm = np.zeros((X.size,Y.size))

# Solve B-field
for i in range(0,X.size):
  for j in range(0,Y.size):
    for k in range(0,Ncoils-1):
      Ra     = Loops[k][0]
      I0     = Loops[k][1]
      Nturns = Loops[k][2]
      Center = Loops[k][3:6]
      Angles = Loops[k][6:9]
      Point = np.array([ X[i], Y[j], 1e-10 ])
      Bx,By,Bz = bfield.loopxyz( Ra,I0,Nturns,Center,Angles, Point )
      Bnorm[i][j] += np.sqrt( Bx*Bx + By*By + Bz*Bz )

plt.figure(1)
XX,YY = np.meshgrid(X,Y)
plt.contour(np.transpose(XX),np.transpose(YY),Bnorm,120)
plt.colorbar()
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('B-field magnitude [T] - Simple Toroidal Field')
plt.savefig('ex08_plot_simple_toroidal_field_bnorm.png',dpi=150)
plt.show()

plt.figure(2)
plt.plot(X,Bnorm[:,0]*1e4)
plt.xlim([R0-a0, R0+a0])
plt.ylim([0,500])
plt.xlabel('Radius [m]')
plt.ylabel('B [Gauss]')
plt.title('B-field magnitude along radius [T] - Simple Toroidal Field')
plt.savefig('ex08_plot_simple_toroidal_field_baxis.png',dpi=150)
plt.show()
