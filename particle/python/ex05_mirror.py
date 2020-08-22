import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

sys.path.insert(1, '../../ode/python/')
sys.path.insert(1, '../../bfield/python/')

import ode
import bfield

qe  = 1.60217662e-19
me  = 9.10938356e-31
mp  = 1.6726219e-27
mu0 = 4.0*np.pi*1.0e-7
qm  = qe/mp

I0 = 500.0 # [A]
Ra = 0.1   # [m]
Nt = 100  # turns


def Efield(x,y,z):
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0
    return Ex, Ey, Ez


def Bfield(x,y,z):
    # Major radius (projection)
    r = np.sqrt( x*x + y*y )
    # Sin, Cos of particle position on (x,y) plane
    ca = x/r
    sa = y/r
    # B-field of one loop
    Br, Bz  = bfield.loopbrz(Ra,I0,Nt,r,z)
    Bx = Br*ca
    By = Br*sa
    return Bx, By, Bz


# RHS of the ODE problem, dy/dx = f(x,y)
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
   vy0 =  2.0e5
   vz0 = -2.56e5

   # B-field max [T]
   B0 =  mu0 * I0 * Nt / 2.0 / np.pi / Ra
   # Larmor pulsation [rad/s]
   w_L = B0*qm
   # Larmor period [s]
   tau_L = 2.0*np.pi/w_L
   # Larmor radius [m]
   r_L = vy0/w_L
   # Mirror Ratio
   Rm = ( vy0**2 + vz0**2 )/(vy0**2)
   # Sine of the Loss angle
   sth = np.sqrt( 1.0/Rm )
   # Loss angle
   thm = np.arcsin( sth ) *180.0/np.pi

   # Initial conditions
   X0 = np.array(( 0.001, 0.0, Ra, 0.0, vy0, vz0 ))

   # Number of Larmor gyrations
   N_gyros = 2

   # Time grid
   time = np.linspace( 0.0, tau_L*N_gyros, 75*N_gyros )

   # Runge-Kutta 4
   X = ode.rk4( fun, time, X0 )

   # Collect components
   x  = X[:,0]
   y  = X[:,1]
   z  = X[:,2]
   vx = X[:,3]
   vy = X[:,4]
   vz = X[:,5]

   # Current Loop points
   theta = np.linspace(0.0, 2*np.pi, 100)
   Xloop = Ra * np.cos(theta)
   Yloop = np.zeros(100)
   Zloop = Ra * np.sin(theta)

   # Plot 1 - Trajectory
   plt.figure(1)
   plt.plot( x, z, 'b-', label='orbit' )
   plt.plot( Xloop, Yloop, 'r', label='loop')
   plt.xlabel('x [m]')
   plt.ylabel('z [m]')
   plt.axis('equal')
   plt.legend()

   # Add few fieldlines

   # Initial position of the field line
   Nlines = 10
   fieldlines_X0     = np.linspace( 0, Ra*0.98, Nlines )
   fieldlines_Y0     = np.linspace( 0,   0,  Nlines )
   fieldlines_Z0     = np.linspace( 0,   0,  Nlines )
   fieldlines_direction = np.ones( Nlines )
   fieldlines_length = np.ones( Nlines ) * 0.1
   Center   = np.array([0, 0, 0])
   Uhat     = np.array([0, 1, 0])
   Npoints  = 20
   filament = bfield.makeloop( Ra, Center, Uhat, 100 )

   for i in range(np.size(fieldlines_X0,0)):
       # Top portion
       Y0 = np.array([ fieldlines_X0[i], fieldlines_Y0[i],fieldlines_Z0[i],fieldlines_direction[i]])
       interval = x=np.arange(0.0,fieldlines_length[i],1e-4)
       fieldlines = odeint(bfield.blines, Y0, interval, args=(filament, I0) )
       plt.plot( fieldlines[:,0], fieldlines[:,1], 'k-' )
       # Bottom portion
       Y0 = np.array([ fieldlines_X0[i], fieldlines_Y0[i],fieldlines_Z0[i],-fieldlines_direction[i]])
       interval = x=np.arange(0.0,fieldlines_length[i],1e-4)
       fieldlines = odeint(bfield.blines, Y0, interval, args=(filament, I0) )
       plt.plot( fieldlines[:,0], fieldlines[:,1], 'k-' )
   plt.xlim([0, 1.5*Ra])
   plt.ylim([-0.1*Ra, 1.4*Ra])
   plt.xlabel('X [m]')
   plt.ylabel('Z [m]')
   plt.title('Mirroring of a particle launched toward a current loop')
   plt.savefig('ex05_mirror_trajectory.png',dpi=150)
   plt.show()

   plt.figure(2)
   plt.plot(time*1e6, vz)
   plt.xlabel('time [micro-seconds]')
   plt.ylabel('Vz [m/s]')
   plt.title('Velocity along the axis of the current loop')
   plt.savefig('ex05_mirror_vaxial.png',dpi=150)
   plt.show()


if __name__ == '__main__':
   main()
