import numpy as np
from pylab import plot, axis, show, xlim, ylim, savefig
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
Np = 20

# Charge and Mass
q = np.concatenate( (qe*np.ones(int(Np/2)), -qe*np.ones(int(Np/2)) ) )
m = np.concatenate( (mp*np.ones(int(Np/2)),  me*np.ones(int(Np/2)) ) )

# Characteristic time [s]
T = 200.0*tb #20.0*tb
print('T=',T,' [s]')

# Characteristic size of the domain [m]
L = 100.0*a0
print('L=',L,' [m]')

Rx = np.random.rand(Np)*L
Ry = np.random.rand(Np)*L
Rz = np.random.rand(Np)*L
Vx = np.concatenate( ( np.zeros(int(Np/2)), np.random.rand(int(Np/2))*vb/18.0 ) )
Vy = np.concatenate( ( np.zeros(int(Np/2)), np.random.rand(int(Np/2))*vb/18.0 ) )
Vz = np.concatenate( ( np.zeros(int(Np/2)), np.random.rand(int(Np/2))*vb/18.0 ) )


# Dynamic function, Newton-Lorentz Equation
def dynamics(time,y):

    rx = y[0*Np:1*Np]
    ry = y[1*Np:2*Np]
    rz = y[2*Np:3*Np]

    vx = y[3*Np:4*Np]
    vy = y[4*Np:5*Np]
    vz = y[5*Np:6*Np]

    # Electric field
    Ex = 0.0
    Ey = 0.0
    Ez = 0.0

    # Magnetic field
    Bx = 0.0
    By = 0.0
    Bz = 0.0

    ax = np.zeros(Np)
    ay = np.zeros(Np)
    az = np.zeros(Np)

    for i in range(Np):
       for j in range(Np):
          if (j!=i):

             rx_ij = rx[i] - rx[j]
             ry_ij = ry[i] - ry[j]
             rz_ij = rz[i] - rz[j]

             r_ij = np.sqrt( rx_ij**2 + ry_ij**2 + rz_ij**2 ) + epsilon

             Fx_ij = kc * q[i]*q[j] * rx_ij / (r_ij**3)
             Fy_ij = kc * q[i]*q[j] * ry_ij / (r_ij**3)
             Fz_ij = kc * q[i]*q[j] * rz_ij / (r_ij**3)

             ax[i] += Fx_ij/m[i]
             ay[i] += Fy_ij/m[i]
             az[i] += Fz_ij/m[i]

#       qm = q[i]/m[i]
#
#       ax[i] += qm*Ex + qm*(Bz*vy[i]-By*vz[i])
#       ay[i] += qm*Ey + qm*(Bx*vz[i]-Bz*vx[i])
#       az[i] += qm*Ez + qm*(By*vx[i]-Bx*vy[i])

    return np.concatenate( (vx, vy, vz, ax, ay, az) )


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


# Time interval
tspan = np.linspace(0.0, T, 200)

# Initial conditions
Y0 = np.zeros( (6*Np) )
Y0 = np.concatenate( ( Rx, Ry, Rz, Vx, Vy, Vz ) )

# Solve the ODE
Y = ode4( dynamics, Y0, tspan )

Rx = Y[ :, 0*Np:1*Np ]
Ry = Y[ :, 1*Np:2*Np ]
Rz = Y[ :, 2*Np:3*Np ]

Vx = Y[ :, 3*Np:4*Np ]
Vy = Y[ :, 4*Np:5*Np ]
Vz = Y[ :, 5*Np:6*Np ]

# Plot
plot( Rx[:,0:int(Np/2)], Ry[:,0:int(Np/2)], 'ro-') # protons
plot( Rx[:,int(Np/2):Np], Ry[:,int(Np/2):Np], 'b-') # electrons
xlim([ 0, L ])
ylim([ 0, L ])
savefig('nbody_1.png',dpi=200)
show()
