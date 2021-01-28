import matplotlib.pyplot as plt
import numpy as np
from src import charged_particle
import util

'''
[1] He, Yang, et al. “Volume-Preserving Algorithms for Charged Particle
Dynamics.” Journal of Computational Physics, vol. 281, no. 1, 22 Oct. 2014.

This script is to reproduce example 4.1 in [1]. Specifically, is
is used to produce a 2-D trajectory of a test charge (q/m == 1)
into the following electromagnetic field:

B = R * zhat
E = ( 10^(-2)/R^3 ) * (x * xhat + y*yhat)

where R = sqrt(x^2 + r^2), and xhat, yhat, and zhat are the basis vectors.

The initial position of the test charge is [0, -1, 0], and its initial velocity
is [0.1, 0.01, 0].

When this script is called, either the Boris-Bunemmann OR the Gh2 algorithm is
used to calculate the trajectory (see the main function)
'''

def Bfield_static_radial(x):
    """
    B field described in Section 4.1 of [1]
    """
    R = np.sqrt(x[0]*x[0] + x[1]*x[1])
    Bx = 0.0
    By = 0.0
    Bz = R
    return np.array((Bx, By, Bz))


def Efield_static_radial_divergent(x):
    """
    E field described in Section 4.1 of [1]
    """
    R = np.sqrt(x[0]*x[0] + x[1]*x[1])
    B0 = 1e02
    Ex = B0 * x[0] / R**3
    Ey = B0 * x[1] / R**3
    Ez = 0.0
    return np.array((Ex, Ey, Ez))


def boris_long_e_cross_b_drift():

    # Normalized units
    x0 = 0.0
    y0 = -1
    z0 = 0.0
    vx0 = 0.1
    vy0 = 0.01
    vz0 = 0
    X0 = np.array([x0, y0, z0, vx0, vy0, vz0])

    N_gyro = 260
    h = np.pi / 10.0  # This is the value given in Example 4.1 of [1]
    time = np.arange(
        0.0,
        N_gyro * 2 * np.pi,
        step = h,
    )

    particle = charged_particle.TestCharge(X0)

    # Orbit the electron around the guiding center multiple times,
    # in order to compare different orbits across super-long integrations.
    Nrevolutions = 1
    X = particle.boris_bunemann_trajectory(
        time,
        Efield_static_radial_divergent,
        Bfield_static_radial,
        Nrevolutions = Nrevolutions,
    )

    plt.figure()
    plt.plot(X[:,0], X[:,1])
    plt.title('Larmor Gyration and Guided Center Drive of Test Charge')
    plt.xlabel('X (normalized)')
    plt.ylabel('Y (normalized)')
    plt.show()
    plt.savefig(
        'images/boris_e_cross_b_drift_rev_%d.png' % Nrevolutions,
        dpi = 150,
    )


def gh2_long_e_cross_b_drift():

    # Normalized units
    x0 = 0.0
    y0 = -1
    z0 = 0.0
    vx0 = 0.1
    vy0 = 0.01
    vz0 = 0
    X0 = np.array([x0, y0, z0, vx0, vy0, vz0])

    N_gyro = 260
    h = np.pi / 10.0  # This is the value given in Example 4.1 of [1]
    time = np.arange(
        0.0,
        N_gyro * 2 * np.pi,
        step = h,
    )

    particle = charged_particle.TestCharge(X0)
    Nrevolutions = 1
    X = particle.Gh2_trajectory(
        time,
        Efield_static_radial_divergent,
        Bfield_static_radial,
        Nrevolutions = Nrevolutions,
    )

    plt.figure()
    plt.plot(X[:,0], X[:,1])
    plt.title('Larmor Gyration and Guided Center Drive of Test Charge')
    plt.xlabel('X (normalized)')
    plt.ylabel('Y (normalized)')
    plt.show()
    plt.savefig(
        'images/gh2_e_cross_b_drift_rev_%d.png' % Nrevolutions,
        dpi = 150,
    )


if __name__ == '__main__':
    util.mkdir('images')
    #boris_long_e_cross_b_drift()
    gh2_long_e_cross_b_drift()
