import ode
import numpy as np
import matplotlib.pyplot as plt

def fun(t,y):
    ydot = y - t**2 + 1
    return ydot

def main():
    tn   = np.linspace( 0.0, 5.0, 20 )     # Grid
    y0   = np.array( [ 0.5 ] )            # Initial condition
    #y_ef = ode.euler( fun, tn, y0 )       # Forward Euler
    #y_mp = ode.midpoint( fun, tn, y0 )    # Explicit Midpoint
    y_rk = ode.rk4( fun, tn, y0 )         # Runge-Kutta 4
    y_k3 = ode.k3( fun, tn, y0) # Kutta 3rd order
    y_h3 = ode.heun3( fun, tn, y0) # Heun 3rd order
    y_ral3 = ode.ralston3( fun, tn, y0) # Ralston 3rd order
    y_an = tn**2 + 2.0*tn + 1.0 - 0.5*np.exp(tn) # Analytical

    y_err = np.empty_like(y_an)

    for i in range(len(tn)):
        y_err[i] = ode.error_absolute(y_an[i],y_rk[i])

    plt.figure(1)
    #plt.plot( tn, y_ef, 'ro-', label='Forward Euler (1st)' )
    #plt.plot( tn, y_mp, 'go-', label='Explicit Mid-Point (2nd)' )
    #plt.plot( tn, y_rk, 'bx-', label='Runge-Kutta (4th)' )
    plt.plot( tn, y_an, 'b-',  label='Analytical Solution' )
    plt.plot( tn, y_k3, 'ro-',  label='Kutta 3rd order' )
    plt.plot( tn, y_h3, 'go-',  label='Heun 3rd order' )
    plt.plot( tn, y_ral3, 'kx-',  label='Ralston 3rd order' )
    #plt.plot( tn, y_err, 'kd-',  label='Absolute error' )
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend(loc=3)
    plt.savefig('ex01_ode_new_schemes.png')
    plt.show()

if __name__ == '__main__':
   main()
