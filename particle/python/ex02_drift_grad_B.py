import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../../ode/python/')
import ode

qe = 1.60217662e-19
me = 9.10938356e-31
mp = 1.6726219e-27

# Charge-to-mass ratio (q/m)
qm = qe/mp


def Efield(x,y,z):
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0
    return Ex, Ey, Ez


def Bfield(x,y,z):
    Bx = 0.0
    By = 0.0
    Bz = 0.5*(1.0 + 50.0*x)
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

    # Initial velocity [m/s]
    vy0 = 1.0e5

    # Magnetic Field at the origin [T]
    Bx, By, Bz = Bfield(0,0,0)

    # Larmor pulsation [rad/s]
    w_L = np.abs(qm * Bz)

    # Larmor period [s]
    tau_L = 2.0*np.pi / w_L

    # Larmor radius [m]
    r_L = vy0 / w_L

    # Initial conditions
    X0 = np.array( [ r_L, 0.0, 0.0, 0.0, vy0, 0.0 ] )

    # Number of Larmor gyrations
    N_gyro = 20

    # Number of points per gyration
    N_points_per_gyration = 400

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

    # Perpendicular Kinetic Energy [eV]
    E_kin_perp = 0.5 * mp * ( vx*vx + vy*vy ) / qe

    # Magnetic field along the trajectory
    Bnorm = np.zeros(x.size)
    for i in range(x.size):
        Bx, By, Bz = Bfield(x[i], y[i], z[i])
        Bnorm[i] = np.sqrt(Bx*Bx + By*By + Bz*Bz)

    # Magnetic moment along the trajectory [eV/T]
    mu = np.zeros(x.size)
    for i in range(x.size):
        mu[i] = E_kin_perp[i] / Bnorm[i]

    plt.figure(1)
    plt.plot( X[:,0], X[:,1], 'b-', label='Runge-Kutta (4th)' )
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.legend(loc=3)
    plt.savefig('ex02_drift_grad_B_trajectory.png')
    plt.show()

    plt.figure(2)
    plt.plot( time*1e6, mu )
    plt.xlabel('time [micro-seconds]')
    plt.ylabel('Magnetic Moment [eV/T]')
    plt.savefig('ex02_drift_grad_B_magnetic_moment.png')
    plt.show()


if __name__ == '__main__':
   main()
