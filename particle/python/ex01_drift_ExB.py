import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../../ode/python/')
import ode

qe = 1.60217662e-19
me = 9.10938356e-31
mp = 1.6726219e-27

def fun(t,X):
   x, y, z, vx, vy, vz = X
   # Charge-to-mass ratio (q/m)
   qm = -qe/me
   # E-field [V/m]
   Ex = 0.0
   Ey = 100.0
   Ez = 0.0
   # B-field [T]
   Bx = 0.0
   By = 0.0
   Bz = 1.0e-4
   # Newton-Lorentz equation in Cartesian coordinates
   Xdot = np.zeros(6)
   Xdot[0] = vx
   Xdot[1] = vy
   Xdot[2] = vz
   Xdot[3] = qm * ( Ex + vy*Bz - vz*By )
   Xdot[4] = qm * ( Ey + vz*Bx - vx*Bz )
   Xdot[5] = qm * ( Ez + vx*By - vy*Bx )
   return Xdot


def main():
    # Grid
    time = np.linspace( 0.0, 1.1e-6, 100 )
    # Initial conditions
    X0 = np.array(( 0.0, 0.0, 0.0, 0.0, 1.0e6, 0.0 ))
    # Solve ODE
    X_ef = ode.euler( fun, time, X0 )       # Forward Euler
    X_mp = ode.midpoint( fun, time, X0 )    # Explicit Midpoint
    X_rk = ode.rk4( fun, time, X0 )         # Runge-Kutta 4

    # for i in range(0,xn.size):
    #     print xn[i], y_an[i], y_ef[i,0], y_mp[i,0], y_rk[i,0]

    plt.figure(1)
    plt.plot( X_ef[:,0], X_ef[:,1], 'ro-', label='Forward Euler (1st)' )
    plt.plot( X_mp[:,0], X_mp[:,1], 'go-', label='Explicit Mid-Point (2nd)' )
    plt.plot( X_rk[:,0], X_rk[:,1], 'bx-', label='Runge-Kutta (4th)' )
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.legend(loc=3)
    plt.savefig('ex01_particle_ExB.png')
    plt.show()

if __name__ == '__main__':
   main()
