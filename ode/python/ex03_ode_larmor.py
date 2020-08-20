import ode
import numpy as np
import matplotlib.pyplot as plt

qe = 1.60217662e-19
me = 9.10938356e-31
B0 = 0.1

# RHS of the ODE problem, dy/dx = f(x,y)
def fun(t,y):
   vx = y[3]
   vy = y[4]
   vz = y[5]
   # Charge-to-mass ratio (q/m)
   qm = -qe/me
   # E-field [V/m]
   Ex = 0.0
   Ey = 0.0
   Ez = 0.0
   # B-field [T]
   Bx = 0.0
   By = 0.0
   Bz = B0
   # Newton-Lorentz equation (in Cartesian Coords)
   ax = qm * Ex + qm*( Bz*vy - By*vz )
   ay = qm * Ey + qm*( Bx*vz - Bz*vx )
   az = qm * Ez + qm*( By*vx - Bx*vy )
   ydot = np.array(( vx, vy, vz, ax, ay, az ))
   return ydot


def main():

   # Initial velocity [m/s]
   vy0 = 1.0e6
   # Larmor pulsation [rad/s]
   w_L = qe/me * B0
   # Larmor period [s]
   tau_L = 2.0*np.pi/w_L
   # Larmor radius [m]
   r_L = vy0/w_L

   # Initial conditions
   y0 = np.array(( r_L, 0.0, 0.0, 0.0, vy0, 0.0 ))

   # Euler Forward
   time_ef = np.linspace( 0.0, tau_L, 10 )
   y_ef = ode.euler( fun, time_ef, y0 )

   # Explicit Midpoint
   time_mp = np.linspace( 0.0, tau_L, 10 )
   y_mp = ode.midpoint( fun, time_mp, y0 )

   # Runge-Kutta 4
   time_rk = np.linspace( 0.0, tau_L, 10 )
   y_rk = ode.rk4( fun, time_rk, y0 )

   # Amplitude (orbit radius)
   r_ef = np.sqrt( y_ef[:,0]**2 + y_ef[:,1]**2 )
   r_mp = np.sqrt( y_mp[:,0]**2 + y_mp[:,1]**2 )
   r_rk = np.sqrt( y_rk[:,0]**2 + y_rk[:,1]**2 )

   # Plot 1 - Trajectory
   plt.figure(1)
   plt.plot( y_ef[:,0], y_ef[:,1], 'ro-', label='Euler-Forward (1st)' )
   plt.plot( y_mp[:,0], y_mp[:,1], 'go-', label='Explicit Mid-Point (2nd)' )
   plt.plot( y_rk[:,0], y_rk[:,1], 'bx-', label='Runge-Kutta (4th)' )
   plt.axis('equal')
   plt.xlabel('x [m]')
   plt.ylabel('y [m]')
   plt.title('One Larmor Gyration')
   plt.legend(loc=3)
   plt.savefig('ex03_ode_larmor_trajectory.png')

   # Plot 2 - Amplitude percent error
   plt.figure(2)
   plt.plot( time_ef/tau_L, ode.error_percent( r_L, r_ef), 'ro', label='Forward Euler (1st)' )
   plt.plot( time_mp/tau_L, ode.error_percent( r_L, r_mp), 'go', label='MidPoint (2nd)' )
   plt.plot( time_rk/tau_L, ode.error_percent( r_L, r_rk), 'bx', label='Runge-Kutta (4th)' )
   plt.xlabel('time / tau_Larmor')
   plt.ylabel('Percent Amplitude error [%]')
   plt.title('Percent Amplitude Error over 1 Larmor gyration ')
   plt.legend(loc=2)
   plt.savefig('ex03_ode_larmor_error.png')
   plt.show()


if __name__ == '__main__':
   main()
