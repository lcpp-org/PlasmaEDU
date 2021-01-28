import matplotlib.pyplot as plt
import numpy as np
from src import charged_particle
import util

'''
[1] He, Yang, et al. “Volume-Preserving Algorithms for Charged Particle
Dynamics.” Journal of Computational Physics, vol. 281, no. 1, 22 Oct. 2014.

This script is to reproduce example 4.3 in [1]. Specifically, is
is used to produce a 2-D trajectory (r and z coordinates) of a test charge
(q/m == 1) in a static B field (E field is 0). The B field is given by:

Bx = (2y + xz)/2R^2

By = (2x - yz)/2R^2

Bz = (R - 1)/2R

where R = sqrt(x^2 + r^2).

The initial position of the test charge is [1.05, 0, 0], and its initial velocity
is [0, 4.816E-4, -2.059E-3].

When this script is called, either the Boris-Bunemmann OR the Gh2 algorithm is
used to calculate the trajectory (see the main function)
'''

def Bfield_tokamak(X):
    '''
    B-field described in section 4.2 of [1]
    '''
    x, y, z  = X[:3]
    R = np.sqrt(x*x + y*y)
    Bx = (-2*y - x*z)/(2*R*R)
    By = (2*x - y*z)/(2*R*R)
    Bz = (R-1)/(2*R)
    return np.array((Bx, By, Bz))


def Efield_tokamak(X):
    '''
    E-field described in section 4.1 of [1]
    '''
    return np.zeros(3)


def boris_on_banana_orbit():

    # Normalized units
    x0 = 1.05
    y0 = 0
    z0 = 0
    vx0 = 0
    vy0 = 4.816e-4
    vz0 = -2.059e-3

    X0 = np.array((x0,y0,z0,vx0,vy0,vz0))

    # Time grid (s)
    N_gyro = 1e4

    dt = np.pi/ 10.0
    time = np.arange(
        0.0,
        N_gyro * 2 * np.pi,
        step = dt,
    )

    particle = charged_particle.TestCharge(X0)

    print('computing %d periods for Boris Tokamak trajectory...' % N_gyro)
    X = particle.boris_bunemann_trajectory(
        time,
        Efield_tokamak,
        Bfield_tokamak,
    )
    print('done')

    plt.figure(1)
    R = np.sqrt(X[:,0]**2 + X[:,1]**2)
    plt.plot(R, X[:,2], 'b.-')
    plt.xlabel('$R$')
    plt.ylabel('$z$')
    plt.title(
        'Banana Orbit in a 2D Axisymmetric Tokamak B-Field with'
        + ' no E-Field'
    )
    plt.grid()
    #plt.show()
    plt.savefig('images/boris_banana.png')


def gh2_on_banana_orbit():

    # Normalized units
    x0 = 1.05
    y0 = 0
    z0 = 0
    vx0 = 0
    vy0 = 4.816e-4
    vz0 = -2.059e-3

    X0 = np.array((x0,y0,z0,vx0,vy0,vz0))

    # Time grid (s)
    N_gyro = 1e4

    dt = np.pi/ 10.0
    time = np.arange(
        0.0,
        N_gyro * 2 * np.pi,
        step = dt,
    )

    particle = charged_particle.TestCharge(X0)
    print('computing %d periods for a Gh2 Tokamak trajectory...' % N_gyro)
    X = particle.Gh2_trajectory(
        time,
        Efield_tokamak,
        Bfield_tokamak,
    )
    print('done')

    plt.figure(1)
    R = np.sqrt(X[:,0]**2 + X[:,1]**2)
    plt.plot(R, X[:,2], 'b.-')
    plt.xlabel('$R$')
    plt.ylabel('$z$')
    plt.title(
        'Banana Orbit in a 2D Axisymmetric Tokamak B-Field with'
        + ' no E-Field'
    )
    plt.grid()
    #plt.show()
    plt.savefig('images/gh2_banana.png')


if __name__ == '__main__':
    util.mkdir('images')
    #boris_on_banana_orbit()
    gh2_on_banana_orbit()
