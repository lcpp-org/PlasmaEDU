import ode
import numpy as np
import matplotlib.pyplot as plt

def fun(x,y):
    ydot = y - x*x + 1.0
    return ydot

def main():
    xn   = np.linspace( 0.0, 5.0, 20 )     # Grid
    y0   = np.array( [ 0.5 ] )            # Initial condition
    y_ef = ode.euler( fun, xn, y0 )       # Forward Euler
    y_mp = ode.midpoint( fun, xn, y0 )    # Explicit Midpoint
    y_rk = ode.rk4( fun, xn, y0 )         # Runge-Kutta 4
    y_an = xn**2 + 2.0*xn + 1.0 - 0.5*np.exp(xn) # Analytical                     # Analytical

    for i in range(0,xn.size):
        print xn[i], y_an[i], y_ef[i,0], y_mp[i,0], y_rk[i,0]

    plt.figure(1)
    plt.plot( xn, y_ef, 'ro-', label='Forward Euler (1st)' )
    plt.plot( xn, y_mp, 'go-', label='Explicit Mid-Point (2nd)' )
    plt.plot( xn, y_rk, 'bx-', label='Runge-Kutta (4th)' )
    plt.plot( xn, y_an, 'k-',  label='Analytical Solution' )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=3)
    plt.savefig('ex02_ode_solution.png')
    plt.show()

if __name__ == '__main__':
   main()
