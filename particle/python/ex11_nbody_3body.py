import numpy as np
from pylab import plot, axis, show, savefig
from scipy.integrate import odeint

# Physical Constants (SI units, 2019 redefinition)
qe   = 1.602176634e-19       # fundamental charge [C]
me   = 9.109383701528e-31    # electron rest mass [kg]
mp   = 1.6726219236951e-27   # proton rest mass [kg]
lux  = 299792458.0           # speed of light [m/s]
hp   = 6.62607015e-34        # Planck constant [Js]
muref= 1.0000000005415e-7    # Reference measure of mu0
mu0  = 4.0*np.pi*muref       # Vacuum permeability [H/m]
eps0 = 1/lux/lux/mu0         # Vacuum permittivity [F/m]
fine = qe*qe*lux*mu0/2.0/hp  # Fine structure
kc   = 1.0/4.0/np.pi/eps0    # Coulomb constant
hbar = hp/2.0/np.pi          # h-bar
epsilon = 1.0e-15            # Small number (nucleus size) [m]

# Bohr model (SI units)
a0   = hbar/me/lux/fine             # Bohr radius
mk   = kc*qe*qe/me
vb   = np.sqrt(mk/a0)               # Bohr speed
tb   = 2.0*np.pi*np.sqrt(a0**3/mk)  # Bohr period

# Number of particles
Np = 3

# Charge and mass
q  = np.array( [ qe, -qe, -qe  ] )
m  = np.array( [ mp,  me,  me  ] )

def ode4( f, y0, x ):
   '''
    Runge-Kutta 4th order
    -----------------------------
    Butcher Table:

    0   | 0     0     0     0
    1/2 | 1/2   0     0     0
    1/2 | 0     1/2   0     0
    1   | 0     0     1     0
    -----------------------------
        | 1/6   1/3   1/3   1/6
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * f( x[n]       , y[n,:]        )
      k2 = h * f( x[n]+h/2.0 , y[n,:]+k1/2.0 )
      k3 = h * f( x[n]+h/2.0 , y[n,:]+k2/2.0 )
      k4 = h * f( x[n]+h     , y[n,:]+k3     )
      y[n+1,:] = y[n,:] + k1/3.0 + k2/6.0 + k3/6.0 + k4/3.0
   return y


# Dynamic function, Newton-Lorentz Equation
def dynamics(Y,t):
#def dynamics(t,Y):

    x = Y[0*Np:1*Np]
    y = Y[1*Np:2*Np]
    z = Y[2*Np:3*Np]

    vx = Y[3*Np:4*Np]
    vy = Y[4*Np:5*Np]
    vz = Y[5*Np:6*Np]

    ax = np.zeros(Np)
    ay = np.zeros(Np)
    az = np.zeros(Np)

    for i in range(Np):
       for j in range(Np):
          if (j!=i):

             x_ij = x[i] - x[j]
             y_ij = y[i] - y[j]
             z_ij = z[i] - z[j]

             r_ij = np.sqrt( x_ij**2 + y_ij**2 + z_ij**2 )

             Fx_ij = kc * q[i]*q[j] * x_ij / (r_ij**3)
             Fy_ij = kc * q[i]*q[j] * y_ij / (r_ij**3)
             Fz_ij = kc * q[i]*q[j] * z_ij / (r_ij**3)

             ax[i] += Fx_ij/m[i]
             ay[i] += Fy_ij/m[i]
             az[i] += Fz_ij/m[i]

    dY = np.concatenate( (vx, vy, vz, ax, ay, az) )
    return dY


def main():

    # We assume Particle (1), the proton,
    # intially at rest at the origin of the ref. frame


    # Atomic electron (2)

    Rx2 = a0
    Ry2 = 0.0
    Rz2 = 0.0

    Vx2 = 0.0
    Vy2 = vb
    Vz2 = 0.0


    # Impacting electron (3)

    Rx3 = -10*a0
    Ry3 = a0
    Rz3 = 0.0

    Energy_eV = 30.0  # 10.0, 13.6, 30.0, 5000.0, 10000.0
    Vx3 = np.sqrt(2.0*Energy_eV*qe/me)
    Vy3 = 0.0
    Vz3 = 0.0

    Rx = np.array( [ 0.0, Rx2, Rx3 ] )
    Ry = np.array( [ 0.0, Ry2, Ry3 ] )
    Rz = np.array( [ 0.0, Rz2, Rz3 ] )
    Vx = np.array( [ 0.0, Vx2, Vx3 ] )
    Vy = np.array( [ 0.0, Vy2, Vy3 ] )
    Vz = np.array( [ 0.0, Vz2, Vz3 ] )

    Y0 = np.concatenate( ( Rx, Ry, Rz, Vx, Vy, Vz ) )

    # Time grid
    tspan = np.linspace(0.0, 3.0*tb, 1000)

    # Solve ODE
    Y = odeint(dynamics, Y0, tspan)

    Rx = Y[ :, 0*Np:1*Np ]
    Ry = Y[ :, 1*Np:2*Np ]
    Rz = Y[ :, 2*Np:3*Np ]

    Vx = Y[ :, 3*Np:4*Np ]
    Vy = Y[ :, 4*Np:5*Np ]
    Vz = Y[ :, 5*Np:6*Np ]

    # Plot results
    plot( Rx[:,0], Ry[:,0], 'ro-')
    plot( Rx[:,1], Ry[:,1], 'b-')
    plot( Rx[:,2], Ry[:,2], 'k-')
    axis('equal')
    savefig('nbody_3body.png',dpi=200)
    show()

if __name__ == '__main__':
    main()
