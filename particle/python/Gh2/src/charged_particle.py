import numpy as np
import matplotlib.pyplot as plt

'''
[1] Hockney, Roger W. Computer Simulation Using Particles. CRC Press, 1981.


[2] He, Yang, et al. “Volume-Preserving Algorithms for Charged Particle
Dynamics.” Journal of Computational Physics, vol. 281, no. 1, 22 Oct. 2014.

'''

class ChargedParticle:
    """
    ChargedParticle implements various time-integration
    schemes to solve for the trajectory of a particle in the Newton-Lorentz
    problem.

    There is the option to integrat the trajectory of a charged particle with
    the Boris-Bunemann Scheme, or the "Gh2" algorithm described in [2]
    """

    def __init__(self, X0, q, m):
        self.qm = q/m
        self.X = X0

    def _init_time_integration(self,time):
        N = np.size(time)
        M = np.size(self.X)

        timeseries = np.zeros((N,M))

        timeseries[0] = self.X

        # Assume a linear time step
        dt = time[1] - time[0]

        return timeseries, dt, N

    def boris_bunemann_trajectory(self, time, Efield, Bfield, Nrevolutions = 1):

        timeseries, dt, N = self._init_time_integration(time)

        # (q/m) * (dt/2)
        qmdt2 = self.qm*dt/2.0

        # Since boris evaluates at half time steps, we need to start
        # by pushing back the velocity back by one half time step. We
        # leave the initial position unchanged.
        self.boris_bunemann_step(
            dt, -qmdt2*0.5, Efield, Bfield, skip_pos = True,
        )

        for r in range(Nrevolutions):
            if Nrevolutions > 1:
                print('revolution %d' % r)

            # Overwrite the previous data every revolution
            for n in range(0,N-1):
                self.boris_bunemann_step(dt, qmdt2, Efield, Bfield)

                # Store results
                timeseries[n+1] = self.X

        return timeseries

    def boris_bunemann_step(self, dt, qmdt2, Efield, Bfield, skip_pos = False):
        x, v = self.X[:3], self.X[3:]
        E = Efield(x)
        B = Bfield(x)
        # Find frequency correction
        Bmag = np.sqrt(np.dot(B,B))
        Bhat = B / Bmag
        # Frequency correction. dt_omega2 = (q|B|dt)/2m
        dt_omega2 = np.tan(qmdt2 * Bmag)
        alpha = self.frequency_correction(qmdt2 * B)

        #h_omega = np.tan(-0.5 * self.qm * Bmag * dt)*2
        # Half Acceleration due to E-field
        v += qmdt2 * E * alpha

        # Half Rotation due to B-field
        t = qmdt2 * B * alpha
        tsq = np.dot(t,t)
        s = 2/(1 + tsq) * t
        vprime = v + np.cross(v, t)
        v += np.cross(vprime, s)
        # Half Acceleration due to E-field
        v += qmdt2 * E * alpha

        # Push position
        if not skip_pos:
            x += dt*v

        self.X = np.concatenate((x,v))

    @staticmethod
    def frequency_correction(x):
        alpha = [np.tan(k) / k if k != 0 else 0 for k in x]
        return np.array(alpha)

    def Gh2_trajectory(
            self,
            time,
            Efield,
            Bfield,
            Nrevolutions = 1,
            linear_field_correction = False):
        timeseries, dt, N = self._init_time_integration(time)

        for i in range(Nrevolutions):

            if Nrevolutions > 1:
                print('revolution %d' % i)

            # Overwrite previous data once per revolution
            for n in range(0,N-1):
                self.Gh2_step(dt, Efield, Bfield, linear_field_correction)

                # Store results
                timeseries[n+1] = self.X

        return timeseries

    def Gh2_step(self, dt, Efield, Bfield, linear_field_correction):
        x, v = self.X[:3], self.X[3:]
        qmdt2 = self.qm *dt/2.0

        # Half position push
        xhalf = x + dt*v / 2.0
        E = Efield(xhalf)
        B = Bfield(xhalf)

        Bmag = np.sqrt(np.dot(B,B))
        Bhat = B / Bmag

        # Frequency correction Type 1.
        # These frequency correction are used for examples 4.1 and
        # 4.2 of [2]. They are the only way that I can get the
        # trajectory to look like Boris for both the guided-center
        # ExB field and the Tokamak banana orbit.
        h_omega = -self.qm * Bmag * dt
        alpha = self.frequency_correction(self.qm * B * dt / 2.0)

        # Frequency correction Type 2. This is the only frequency
        # correction I was able to use for a Larmor gyration in
        # a static, uniform E field (0.0, 0.0, 0.1) and B field
        # (0.0, 0.0, 0.1) which produce bounded errors for long
        # integrations. However, using these corrections, the
        # trajectory for the ExB field (Example 4.1) and Tokamak
        # (Example 4.2) was indiscernable and obviously incorrect.
        # It could have something to do with the paragraph immediately
        # following Equation 4-77 of Chapter 4, page 110 of [1]
        # ("Unfortunately, for forces which are nonlinear functions of x,
        # the frequency correction cannot be so simply made"). Here,
        # I am directly swapping h_omega with h_omega_tilde described
        # after Equation 11 of [2].
        if linear_field_correction:
            h_omega = 2*np.tan(-0.5 * self.qm * Bmag * dt)
            alpha = 1.0

        # Half Acceleration due to E-field
        v += qmdt2 * E * alpha

        # Rotation due to B-field
        b_crossv = np.cross(Bhat, v)
        v1 = (4*h_omega/(4 + h_omega**2))*b_crossv
        v2 = (2*h_omega**2)/(4 + h_omega**2)*np.cross(Bhat, np.cross(Bhat, v))
        v += v1 + v2

        # Half Acceleration due to E-field
        v += qmdt2 * E * alpha

        # Push position
        x = xhalf + dt * v / 2.0

        self.X = np.concatenate((x,v))


class Electron(ChargedParticle):
    Q = -1.60217662e-19  # Coulombs
    M = 9.10938356e-31   # Kg

    def __init__(self, X0):
        ChargedParticle.__init__(self, X0, self.Q, self.M)


class Proton(ChargedParticle):
    Q = 1.60217662e-19  # Coulombs
    M = 1.67262192e-27  # Kg

    def __init__(self, X0):
        ChargedParticle.__init__(self, X0, self.Q, self.M)


class TestCharge(ChargedParticle):
    """
    Test Charge is a useful object for normalized integration schemes
    """
    Q = 1.0
    M = 1.0

    def __init__(self, X0):
        ChargedParticle.__init__(self, X0, self.Q, self.M)


def Bfield_static_uniform(x):
    """
    To simplify the example, impose a constant, uniform B field
    """
    Bx = 0.0
    By = 0.0
    Bz = 0.1
    return np.array((Bx, By, Bz))


def Efield_static_uniform(x):
    """
    To simplify the example, impose a constant, uniform E field
    """
    Ex = 0
    Ey = 0
    Ez = 0.1
    return np.array((Ex, Ey, Ez))


def analytic_larmor_gyration(time, Tc, r_L):
    """
    This function is useful for evaluating the error of
    a time integration scheme.
    """
    N = np.size(time)
    X = np.zeros((N,2))

    for i in range(N):
        radians = time[i] * 2 * np.pi / Tc
        x = np.cos(radians) * r_L
        y = np.sin(radians) * r_L
        X[i][0] = x
        X[i][1] = y

    return X


def init_larmor_gyration(v0, q, m, Nsteps_per_period, Nperiods):
    # Need initial B-field to compute Larmor radius
    Bx, By, Bz = Bfield_static_uniform(None)
    Bmag = np.sqrt(Bx*Bx + By*By + Bz*Bz)

    # Cyclotron frequency
    omega_c = np.abs(q)*Bmag/m

    # Cyclotron period
    Tc = 2.0*np.pi/omega_c

    # Larmor radius (m)
    v0Mag= np.sqrt(np.dot(v0,v0))
    r_L = v0Mag/omega_c

    # Initial position (m). Always start one larmor radius for simple
    # trajectory viewing.
    x0 = r_L
    y0 = 0.0
    z0 = 0.0

    # Time grid (s)
    time = np.linspace(0.0, Tc*Nperiods, Nsteps_per_period*Nperiods)

    # Initial conditions
    X0 = np.array([x0, y0, z0, v0[0], v0[1], v0[2]])
    return time, X0, r_L, Tc


def initial_velocity_for_checks():
    """
    The initial velocity to use for sanity-check time integrations.
    """
    # m/s
    vx0 = 0.0
    vy0 = 1e6
    vz0 = 0.0
    v0 = np.array((vx0, vy0, vz0))
    return v0


FIGURE_NUM = -1
def increment_figure():
    global FIGURE_NUM
    FIGURE_NUM += 1
    plt.figure(FIGURE_NUM)


def boris_simple_larmor_check(Nsteps_per_period = 15):
    """
    This is meant to be a sanity check to see whether the integration
    scheme produces the expected circular trajectory of a charged particle
    with initial velocity 1 = (0,1e6,0), uniform Bfield = Efield = (0,0,0.1)
    """
    v0 = initial_velocity_for_checks()

    # Just a single Larmor period here. This is meant to be a sanity check
    # to see whether the integration scheme produces the expected circlular
    # trajectory..
    Nperiods = 1
    time, X0, r_L, _ = init_larmor_gyration(
        v0, Electron.Q, Electron.M, Nsteps_per_period, Nperiods,
    )
    particle = Electron(X0)

    X = particle.boris_bunemann_trajectory(
        time, Efield_static_uniform, Bfield_static_uniform,
    )

    increment_figure()
    plt.title('Single Larmor Period')
    plt.plot(
        X[:,0]/r_L,
        X[:,1]/r_L,
        'b.-',
    )
    plt.ylabel('Y ($r_L$)')
    plt.xlabel('X ($r_L$)')
    plt.show()


def boris_long_integration_error_check(Nsteps_per_period = 15):
    """
    This is meant to test the boris scheme after it has succesfully
    passed boris_simple_larmor_check. The goal is to ensure that the
    error of the Boris-Bunemann trajectory remains low for long
    integrations.
    """
    v0 = initial_velocity_for_checks()

    Nperiods = 400
    time, X0, r_L, Tc = init_larmor_gyration(
        v0, Electron.Q, Electron.M, Nsteps_per_period, Nperiods,
    )
    particle = Electron(X0)

    print('Computing Boris trajectory for %d periods...' % Nperiods)
    X = particle.boris_bunemann_trajectory(
        time, Efield_static_uniform, Bfield_static_uniform,
    )
    print('done')

    analytic_s = analytic_larmor_gyration(time, Tc, r_L)
    increment_figure()
    plt.title('Absolute Error of Boris x(t) for a Simple Larmor Gyration')

    # Since there are a lot of data points, only plot one point per period
    plt.plot(
        time[::Nsteps_per_period-1]/Tc,
        np.abs(X[:,0] - analytic_s[:,0])[::Nsteps_per_period-1]/r_L,
        'b.-',
    )
    plt.ylabel(' err($r_L$)')
    plt.xlabel(' time($t\omega/2\pi$)')
    plt.show()


def gh2_simple_larmor_check(Nsteps_per_period = 15):
    """
    This is meant to be a sanity check to see whether the integration
    scheme produces the expected circular trajectory of a charged particle
    with initial velocity 1 = (0,1e6,0), uniform Bfield = Efield = (0,0,0.1)
    """
    v0 = initial_velocity_for_checks()

    # Just a single Larmor period here. This is meant to be a sanity check
    # to see whether the integration scheme produces the expected circlular
    # trajectory..
    Nperiods = 1
    time, X0, r_L, _ = init_larmor_gyration(
        v0, Electron.Q, Electron.M, Nsteps_per_period, Nperiods,
    )
    particle = Electron(X0)

    X = particle.Gh2_trajectory(
        time,
        Efield_static_uniform,
        Bfield_static_uniform,
        linear_field_correction = True,
    )

    increment_figure()
    plt.title('Single Larmor Period')
    plt.plot(
        X[:,0]/r_L,
        X[:,1]/r_L,
        'b.-',
    )
    plt.ylabel('Y ($r_L$)')
    plt.xlabel('X ($r_L$)')
    plt.show()


def gh2_long_integration_error_check(Nsteps_per_period = 15):
    """
    This is meant to test the boris scheme after it has succesfully
    passed boris_simple_larmor_check. The goal is to ensure that the
    error of the Gh2  trajectory remains low for long
    integrations.
    """
    v0 = initial_velocity_for_checks()

    Nperiods = 400
    time, X0, r_L, Tc = init_larmor_gyration(
        v0, Electron.Q, Electron.M, Nsteps_per_period, Nperiods,
    )
    particle = Electron(X0)

    print('Computing Gh2 trajectory for %d periods...' % Nperiods)
    X = particle.Gh2_trajectory(
        time,
        Efield_static_uniform,
        Bfield_static_uniform,
        linear_field_correction = True,
    )
    print('done')

    analytic_s = analytic_larmor_gyration(time, Tc, r_L)
    increment_figure()
    plt.title('Absolute Error of Boris x(t) for a Simple Larmor Gyration')

    # Since there are a lot of data points, only plot one point per period
    plt.plot(
        time[::Nsteps_per_period-1]/Tc,
        np.abs(X[:,0] - analytic_s[:,0])[::Nsteps_per_period-1]/r_L,
        'b.-',
    )
    plt.ylabel(' err($r_L$)')
    plt.xlabel(' time($t\omega/2\pi$)')
    plt.show()


if __name__ == '__main__':
    """
    Some bare-bones checks for the development process.
    """
    import sys
    _checks = {
        'boris':       boris_simple_larmor_check,
        'boris_long':  boris_long_integration_error_check,
        'gh2':         gh2_simple_larmor_check,
        'gh2_long':    gh2_long_integration_error_check,
    }
    if len(sys.argv) != 2:
        boris_simple_larmor_check()
    check = sys.argv[1].lower()
    if check not in _checks:
        print('\nplease specify one of: ["%s"]' % '", "'.join(_checks.keys()))
        sys.exit(1)
    _checks[check]()

