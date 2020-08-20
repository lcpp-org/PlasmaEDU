import ode
import numpy as np
import matplotlib.pyplot as plt

def fun(t,y):
    ydot = y - t**2 + 1
    return ydot

def main():
    tn   = np.linspace( 0.0, 2.0, 5 )     # Grid
    y0   = np.array( [ 0.5 ] )            # Initial condition
    y_ef = ode.euler( fun, tn, y0 )       # Forward Euler
    y_mp = ode.midpoint( fun, tn, y0 )    # Explicit Midpoint
    y_rk = ode.rk4( fun, tn, y0 )         # Runge-Kutta 4
    y_an = tn**2 + 2.0*tn + 1.0 - 0.5*np.exp(tn) # Analytical

    plt.figure(1)
    plt.plot( tn, y_ef, 'ro-', label='Forward Euler (1st)' )
    plt.plot( tn, y_mp, 'go-', label='Explicit Mid-Point (2nd)' )
    plt.plot( tn, y_rk, 'bx-', label='Runge-Kutta (4th)' )
    plt.plot( tn, y_an, 'k-',  label='Analytical Solution' )
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend(loc=2)
    plt.savefig('ex01_ode_solution.png')
    plt.show()

if __name__ == '__main__':
   main()
