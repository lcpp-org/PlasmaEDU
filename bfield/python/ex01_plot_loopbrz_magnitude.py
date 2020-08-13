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

R = np.linspace(0.001,   0.1, 50 )
Z = np.linspace( -0.05, 0.05, 50 )

loor_Ra    = 0.05
loop_I0    = 100
loop_turns = 1

Bnorm = np.zeros((R.size,Z.size))

for i in range(0,R.size):
  for j in range(0,Z.size):
      Br, Bz = bfield.loopbrz( loor_Ra, loop_I0, loop_turns, R[i], Z[j] )
      Bnorm[i][j] = np.sqrt( Br*Br + Bz*Bz )

plt.figure(1)
RR,ZZ = np.meshgrid(R,Z)
plt.contourf(np.transpose(RR),np.transpose(ZZ),Bnorm,30)
plt.colorbar()
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.title('B-field magnitude [T] of a Current Loop')
plt.savefig('ex01_plot_loopbrz_magnitude.png',dpi=150)
plt.show()
