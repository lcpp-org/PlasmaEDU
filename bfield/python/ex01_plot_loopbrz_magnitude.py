################################################################################
#
#  BFIELD
#
#   Simple example of plot of the magnitude of the magnetic field
#   produced by a current loop on the (R,Z) plane
#
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import bfield

# Current Loop
Ra    = 0.05 # [m] Loop Radius
I0    = 100  # [A] Loop Current
turns = 1    # [#] Number of turns

# R,Z Grid
R = np.linspace(  0.0,  0.1, 50 )
Z = np.linspace(-0.05, 0.05, 50 )

# B-field magnitude
Bnorm = np.zeros((R.size,Z.size))
for i in range(0,R.size):
  for j in range(0,Z.size):
      Br, Bz = bfield.loopbrz( Ra,I0,turns, R[i], Z[j] )
      Bnorm[i][j] = np.sqrt( Br*Br + Bz*Bz )
      print Br,Bz

plt.figure(1)
RR,ZZ = np.meshgrid(R,Z)
plt.contourf(np.transpose(RR),np.transpose(ZZ),Bnorm,30)
plt.colorbar()
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.title('B-field magnitude [T] of a Current Loop')
plt.savefig('ex01_plot_loopbrz_magnitude.png',dpi=150)
plt.show()

# Note on Numpy's contourf([X, Y,] Z, [levels], **kwargs):
# X and Y must both be 2-D with the same shape as Z
# (e.g. created via numpy.meshgrid), or they must both
# be 1-D such that len(X) == M is the number of columns in Z
# and len(Y) == N is the number of rows in Z.
#Zarray-like(N, M) The height values over which the contour is drawn.
