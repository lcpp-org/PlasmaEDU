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

   # Time Grid
   N_gyroperiods = 100
   N_points_per_gyroperiod = 10
   time_rk = np.linspace( 0.0, N_gyroperiods*tau_L, N_gyroperiods*N_points_per_gyroperiod )

   # Runge-Kutta 4
   y_rk = ode.rk4( fun, time_rk, y0 )

   # Amplitude (orbit radius)
   r_rk = np.sqrt( y_rk[:,0]**2 + y_rk[:,1]**2 )

   # Plot 1 - Trajectory
   plt.figure(1)
   plt.plot( y_rk[:,0]/r_L, y_rk[:,1]/r_L, 'b-', label='Runge-Kutta (4th)' )
   plt.axis('equal')
   plt.xlabel('x [r_L]')
   plt.ylabel('y [r_L]')
   plt.title('100 Larmor Gyrations')
   plt.legend(loc=3)
   plt.savefig('ex04_ode_larmor_long_trajectory.png')

   # Plot 2 - Amplitude percent error
   plt.figure(2)
   plt.plot( time_rk/tau_L, ode.error_percent( r_L, r_rk), 'bx', label='Runge-Kutta (4th)' )
   plt.xlabel('time / tau_Larmor')
   plt.ylabel('Percent Amplitude error [%]')
   plt.title('Percent Amplitude Error over 100 Larmor gyrations')
   plt.legend(loc=2)
   plt.savefig('ex04_ode_larmor_long_error.png')
   plt.show()


if __name__ == '__main__':
   main()
