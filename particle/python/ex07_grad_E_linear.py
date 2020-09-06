import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, '../../ode/python/')
import ode

qe = 1.60217662e-19
me = 9.10938356e-31
mp = 1.6726219e-27

# Charge-to-mass ratio (q/m)
qm = -qe/me

# E-field at x=0
E0 = 10.0

# E-field gradient
Eg = 200.0

def Efield(x,y,z):
    Ex = E0 + Eg*x
    Ey = 0.0
    Ez = 0.0
    return Ex, Ey, Ez

def Bfield(x,y,z):
    Bx = 0.0
    By = 0.0
    Bz = 0.0
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


def analytical(t,x0,v0):
    a0 = qm * E0
    a1 = qm * Eg
    rr = E0 / Eg
    w  = np.sqrt(np.absolute(a1))
    wt = w*t
    if (a1>0):
        eps = np.exp(wt)
        c0  = 0.5*eps
        c1  = 0.5/eps
        sum = c0+c1
        dif = c0-c1
        x   = rr*sum - rr + x0*sum + v0/w*dif # 11 FLOPS + 1 EXP
        v   = rr*w*dif + x0*w*dif + v0*sum    #  7 FLOPS
    else:
        cw = np.cos(wt)
        sw = np.sin(wt)
        x  = rr*cw - rr + x0*cw + v0/w*sw # 7 FLOPS + 2 TRIGONOM
        v  = -rr*w*sw - x0*w*sw + v0*cw   # 7 FLOPS
    return x, v


def main():

    # Initial position [m]
    x0 = 0.5
    y0 = 0.0
    z0 = 0.0

    # Initial velocity [m/s]
    vx0 = 1000
    vy0 = 0.0
    vz0 = 0.0

    # Initial conditions
    X0 = np.array( [ x0, y0, z0, vx0, vy0, vz0 ] )

    # Time interval
    T = 5.0e-6

    # Time grid for the RK4 solution
    time_rk4 = np.linspace( 0.0, T, 100 )

    # Solve ODE (Runge-Kutta 4)
    X = ode.rk4( fun, time_rk4, X0 )

    # Get components of the state vector
    x  = X[:,0]
    y  = X[:,1]
    z  = X[:,2]
    vx = X[:,3]
    vy = X[:,4]
    vz = X[:,5]

    # Analytical solution
    time_an = np.linspace( 0.0, T, 1000 )
    x_an = np.zeros(time_an.size)
    v_an = np.zeros(time_an.size)
    for i in range(time_an.size):
      xa, va = analytical(time_an[i], x0, vx0)
      x_an[i] = xa
      v_an[i] = va

    # Characteristic freq
    a1  = qm * Eg
    w   = np.sqrt(np.absolute(a1))
    tau = 2.0*np.pi / w

    plt.figure(1)
    plt.plot( time_an,  x_an, 'k-', label='Analytical Solution' )
    plt.plot( time_rk4, x,    'ro', label='Runge-Kutta (4th)' )
    plt.xlabel('t [s]')
    plt.ylabel('x [m]')
    plt.legend(loc=3)
    plt.savefig('ex07_grad_E_linear_x.png')

    plt.figure(2)
    plt.plot( time_an,  v_an, 'k-', label='Analytical Solution' )
    plt.plot( time_rk4, vx,   'ro', label='Runge-Kutta (4th)' )
    plt.xlabel('t [s]')
    plt.ylabel('v [m/s]')
    plt.legend(loc=3)
    plt.savefig('ex07_grad_E_linear_v.png')
    plt.show()


if __name__ == '__main__':
   main()
