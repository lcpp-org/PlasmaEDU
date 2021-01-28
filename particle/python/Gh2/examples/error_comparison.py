import matplotlib.pyplot as plt
import numpy as np
from src import charged_particle
import util


'''
This script produces a plot comparing the absolute error of the x coordinate
of a test charge in a simple larmor gyration. The error is determined
by comparing the results with the analytic solution (a circle). 

The test charge is placed in a static, uniform electromagnetic field
(E and B both pointing in the +z direction).

The absolute error of the Boris-Bunemann algorithm as well as the Gh2 algorithm
are shown on the same plot.
'''


def boris_gh2_error_comparison(Nsteps_per_period=15):
    """
    Compare the error of Boris and Gh2 on a simple Larmor gyration.
    """
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
    X_boris = particle.boris_bunemann_trajectory(
        time,
        charged_particle.Efield_static_uniform,
        charged_particle.Bfield_static_uniform,
    )

    X_gh2 = particle.Gh2_trajectory(
        time,
        charged_particle.Efield_static_uniform,
        charged_particle.Bfield_static_uniform,
        linear_field_correction = True,
    )

    analytic_s = charged_particle.analytic_larmor_gyration(time, Tc, r_L)
    plt.figure(1)
    plt.title('Absolute Error of X coordinate')
    plt.semilogy(
        time[::Nsteps_per_period - 1]/Tc,
        np.abs(X_gh2[:,0] - analytic_s[:,0])[::Nsteps_per_period - 1]/r_L,
        'b-',
        label =  'Gh2',
    )
    plt.semilogy(
        time[::Nsteps_per_period - 1]/Tc,
        np.abs(X_boris[:,0] - analytic_s[:,0])[::Nsteps_per_period - 1]/r_L,
        'g-',
        label = 'Boris-Bunemann',
    )
    plt.ylabel('err ($r_L$)')
    plt.xlabel('time $t\omega/2\pi$')
    plt.legend(loc=4)
    plt.show()
    plt.savefig('images/boris_gh2_error_comparison.png')


if __name__ == '__main__':
    util.mkdir('images')
    boris_gh2_error_comparison()
