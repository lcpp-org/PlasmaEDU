import matplotlib.pyplot as plt
import numpy as np
from functools import partial

from src import charged_particle
import util



'''
This script produces a plot comparing the absolute error of the x coordinate
of a test charge in a simple larmor gyration. The error is determined
by comparing the results with the analytic solution (a circle). The purpose is
to show how the RK4 algorithm has unbounded error over long time integrations.

The test charge is placed in a static, uniform electromagnetic field
(E and B both pointing in the +z direction).

The absolute error of the Boris-Bunemann algorithm and the RK4 algorithm
are shown on the same plot.
'''

def lorentz_force_derivative(t, X, qm, Efield, Bfield):
    """
    Useful when using generic integration schemes, such
    as RK4, which can be compared to Boris-Bunemann
    """
    v = X[3:]

    E = Efield(X)
    B = Bfield(X)

    # Newton-Lorentz acceleration
    a = qm*E + qm*np.cross(v,B)
    ydot = np.concatenate((v,a))
    return ydot


def rk4_trajectory( time_derivative_func, time, X0 ):
    """
    Classic RK4 (4-stage Butcher tableau)
    """
    N = np.size( time )
    dt = time[1] - time[0]
    M = np.size( X0 )
    X = np.zeros((N,M))
    X[0,:] = X0

    f = time_derivative_func
    for n in range(0, N-1):
        k1 = dt * f( X[n]       , X[n,:]        )
        k2 = dt * f( X[n]+dt/2.0 , X[n,:]+k1/2.0 )
        k3 = dt * f( X[n]+dt/2.0 , X[n,:]+k2/2.0 )
        k4 = dt * f( X[n]+dt     , X[n,:]+k3     )
        X[n+1,:] = X[n,:] + k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0
    return X


def introductory_example(Nsteps_per_period=15):
    vx0 = 0.0
    vy0 = 1e6
    vz0 = 0.0
    v0 = np.array((vx0, vy0, vz0))

    Nperiods = 400
    time, X0, r_L, Tc = charged_particle.init_larmor_gyration(
        v0,
        charged_particle.Electron.Q,
        charged_particle.Electron.M,
        Nsteps_per_period,
        Nperiods,
    )
    particle = charged_particle.Electron(X0)

    print('Computing Boris trajectory for %d periods...' % Nperiods)
    X = particle.boris_bunemann_trajectory(
        time,
        charged_particle.Efield_static_uniform,
        charged_particle.Bfield_static_uniform,
    )

    rk4_derivative_func = partial(
        lorentz_force_derivative,
        qm = charged_particle.Electron.Q/charged_particle.Electron.M,
        Efield = charged_particle.Efield_static_uniform,
        Bfield = charged_particle.Bfield_static_uniform,
    )

    print('Computing RK4 trajectory for %d periods...' % Nperiods)
    X_rk4 = rk4_trajectory(rk4_derivative_func, time, X0)
    print('done')

    analytic_s = charged_particle.analytic_larmor_gyration(time, Tc, r_L)
    plt.figure(1)
    plt.title('Absolute Error of X coordinate')
    plt.semilogy(
        time[::Nsteps_per_period - 1]/Tc,
        np.abs(X_rk4[:,0] - analytic_s[:,0])[::Nsteps_per_period - 1]/r_L,
        'r-',
        label =  'Runge-Kutta 4',
    )
    plt.semilogy(
        time[::Nsteps_per_period - 1]/Tc,
        np.abs(X[:,0] - analytic_s[:,0])[::Nsteps_per_period - 1]/r_L,
        'g-',
        label = 'Boris-Bunemann',
    )
    plt.ylabel('err ($r_L$)')
    plt.xlabel('time $t\omega/2\pi$')
    plt.legend(loc=4)
    plt.show()
    plt.savefig('images/introductory_example.png')


if __name__ == '__main__':
    util.mkdir('images')
    introductory_example()
