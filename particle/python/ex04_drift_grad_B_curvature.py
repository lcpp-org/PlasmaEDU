import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../../ode/python/')
import ode

qe = 1.60217662e-19
me = 9.10938356e-31
mp = 1.6726219e-27
KB = 1.38064852e-23

# Charge-to-mass ratio (q/m)
qm = qe/mp


def Efield(x,y,z):
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0
    return Ex, Ey, Ez


def Bfield(x,y,z):
    B0 = 0.0250
    # Major radius (projection)
    R = np.sqrt( x*x + y*y )
    # Sin, Cos of particle position on (x,y) plane
    ca = x/R
    sa = y/R
    # Toroidal component [T]
    Bphi = B0 * 0.72 / R
    # B-field [T]
    Bx = -Bphi * sa
    By =  Bphi * ca
    Bz =  0.0
    return Bx, By, Bz


def fun(t,X):
   x, y, z, vx, vy, vz = X
   # E-field [V/m]
   Ex, Ey, Ez = Efield(x,y,z)
   # B-field [T]
   Bx, By, Bz = Bfield(x,y,z)
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

    # Thermal speed
    T_eV = 1.0
    v_th = np.sqrt(2*KB*T_eV*11600/mp)

    # Initial velocity [m/s]
    vy0 = v_th
    vz0 = v_th

    # Reference Magnetic Field
    Bx,By,Bz = Bfield(0.72,0,0)
    B0 = np.sqrt(Bx*Bx + By*By + Bz*Bz)

    # Larmor pulsation [rad/s]
    w_L = np.abs(qm * B0)

    # Larmor period [s]
    tau_L = 2.0*np.pi / w_L

    # Larmor radius [m]
    r_L = vy0 / w_L

    # Initial conditions
    X0 = np.array( [ 0.72, 0.0, 0.0, 0.0, vy0, vz0 ] )

    # Number of Larmor gyrations
    N_gyro = 200

    # Number of points per gyration
    N_points_per_gyration = 100

    # Time grid
    time = np.linspace( 0.0, tau_L*N_gyro, N_gyro*N_points_per_gyration )

    # Solve ODE (Runge-Kutta 4)
    X = ode.rk4( fun, time, X0 )

    # Get components of the state vector
    x  = X[:,0]
    y  = X[:,1]
    z  = X[:,2]
    vx = X[:,3]
    vy = X[:,4]
    vz = X[:,5]

    R = np.sqrt(x*x + y*y)

    plt.figure(1)
    plt.plot( x, y, 'b-', label='Runge-Kutta (4th)' )
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.legend(loc=3)
    plt.savefig('ex02_drift_grad_B_trajectory.png')
    plt.show()

    plt.figure(2)
    plt.plot( R, z, 'b-')
    plt.xlabel('R, Radius [m]')
    plt.ylabel('Z, Vertical Coordinate [m]')
    plt.axis('equal')
    plt.savefig('ex02_drift_grad_B_vertical_drift.png')
    plt.show()


if __name__ == '__main__':
   main()
