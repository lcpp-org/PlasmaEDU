################################################################################
#
#  BFIELD
#
#   Simple example of a colormap of the Br,Bz components of the magnetic field
#   produced by a current loop on the (R,Z) plane
#
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import bfield

R = np.linspace(0.001,   0.1, 50 )
Z = np.linspace( -0.05, 0.05, 50 )

loop_I0    = 100
loor_Ra    = 0.05
loop_turns = 1

BR = np.zeros((R.size,Z.size))
BZ = np.zeros((R.size,Z.size))

for i in range(0,R.size):
  for j in range(0,Z.size):
      Br, Bz = bfield.loopbrz( loor_Ra, loop_I0, loop_turns, R[i], Z[j] )
      BR[i][j] = Br
      BZ[i][j] = Bz

plt.figure(1)
RR,ZZ = np.meshgrid(R,Z)
plt.contour(np.transpose(RR),np.transpose(ZZ),BR,30)
plt.colorbar()
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.title('Br, Radial B-field [T] of a Current Loop')
plt.savefig('ex02_plot_loopbrz_components_br.png',dpi=150)

plt.figure(2)
RR,ZZ = np.meshgrid(R,Z)
plt.contour(np.transpose(RR),np.transpose(ZZ),BZ,30)
plt.colorbar()
plt.xlabel('R [m]')
plt.ylabel('Z [m]')
plt.title('Bz, Axial B-field [T] of a Current Loop')
plt.savefig('ex02_plot_loopbrz_components_bz.png',dpi=150)
