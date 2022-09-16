import numpy as np

# Forward Euler 1st order
def euler( fun, x, y0 ):
   '''
    Forward Euler 1st order
    -----------------------------
    Butcher Table:

    0   |  0
    ----------
        |  1
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n], y[n,:] )
      y[n+1,:] = y[n,:] + 1.0*k1
   return y


# Mid-Point 2nd order
def midpoint( fun, x, y0 ):
   '''
    Explicit Mid-point 2nd order
    -----------------------------
    Butcher Table:

    0   | 0     0
    1/2 | 1/2   0
    -----------------
        | 0     1
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]     , y[n,:]        )
      k2 = h * fun( x[n]+h/2 , y[n,:]+k1/2.0 )
      y[n+1,:] = y[n,:] + 0.0*k1 + 1.0*k2
   return y

# Kutta's
def kutta( fun, x, y0 ):
   '''
    Kutta's
    -----------------------------
    Butcher Table:
    0   | 0     0     0
    1/2 | 1/2   0     0
    1   | -1    2     0
    -----------------------------
        | 1/6   2/3   1/6
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]       , y[n,:]                   )
      k2 = h * fun( x[n]+h/2.0 , y[n,:]+k1/2.0           )
      k3 = h * fun( x[n]+h*1.0 , y[n,:]+k1*(-1.0)+k2*2.0 )
      y[n+1,:] = y[n,:] + k1/6.0 + k2*2.0/3.0 + k3/6.0
   return y

# Heun 3rd order
def heun3( fun, x, y0 ):
   '''
    Heun's 3rd order
    -----------------------------
    Butcher Table:
    0   | 0     0     0
    1/3 | 1/3   0     0
    2/3 | 0     2/3   0
    -----------------------------
        | 1/4   0     3/4
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]           , y[n,:]             )
      k2 = h * fun( x[n]+h/3.0     , y[n,:]+k1/3.0     )
      k3 = h * fun( x[n]+h*2.0/3.0 , y[n,:]+k2*2.0/3.0 )
      y[n+1,:] = y[n,:] + k1/4.0 + k3*3.0/4.0
   return y

# Ralston's 3rd order
def ralston3( fun, x, y0 ):
   '''
    Ralston's 3rd order
    -----------------------------
    Butcher Table:
    0   | 0     0     0
    1/2 | 1/2   0     0
    3/4 | 0     3/4   0
    -----------------------------
        | 2/9   1/3   4/9
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]           , y[n,:]            )
      k2 = h * fun( x[n]+h/2.0     , y[n,:]+k1/2.0     )
      k3 = h * fun( x[n]+h*3.0/4.0 , y[n,:]+k2*3.0/4.0 )
      y[n+1,:] = y[n,:] + k1*2.0/9.0 + k2/3.0 + k3*4.0/9.0
   return y

# Runge-Kutta 4th order
def rk4( fun, x, y0 ):
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
      k1 = h * fun( x[n]       , y[n,:]        )
      k2 = h * fun( x[n]+h/2.0 , y[n,:]+k1/2.0 )
      k3 = h * fun( x[n]+h/2.0 , y[n,:]+k2/2.0 )
      k4 = h * fun( x[n]+h     , y[n,:]+k3     )
      y[n+1,:] = y[n,:] + k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0
   return y


def error_absolute( y_reference, y_approx ):
   e_abs = np.abs( y_reference - y_approx )
   return e_abs


def error_relative( y_reference, y_approx ):
   e_rel = np.abs( y_reference - y_approx )/ y_reference
   return e_rel


def error_percent( y_reference, y_approx ):
   e_perc = np.abs( y_reference - y_approx )/ y_reference * 100.0
   return e_perc

# Heun 2nd order
def heun2( fun, x, y0 ):
   '''
    Heun 2nd order
    -----------------------------
    Butcher Table:

    0   | 0     0
    1   | 1     0
    -----------------
        | 1/2   1/2
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]   , y[n,:]   )
      k2 = h * fun( x[n]+h , y[n,:]+k1)
      y[n+1,:] = y[n,:] + k1/2.0 + k2/2.0 
   return y

# Ralston's 2nd order
def ralston2( fun, x, y0 ):
   '''
    Ralston 2nd order
    -----------------------------
    Butcher Table:

    0   | 0     0
    2/3 | 2/3   0
    -----------------
        | 1/4   3/4
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]       , y[n,:]       )
      k2 = h * fun( x[n]+2/3*h , y[n,:]+2/3*k1)
      y[n+1,:] = y[n,:] + k1/4.0 + 3.0*k2/4.0
   return y


def bs( fun, x, y0 ):
   '''
    Bogacki–Shampine 3rd order
    -----------------------------
    Butcher Table:

    0   | 0     0     0     0
    1/2 | 1/2   0     0     0
    3/4 | 0     3/4   0     0
    1   | 2/9   1/3   4/9   0
    -----------------
        | 2/9   1/3   4/9   0
        | 7/24  1/4   1/3   1/8
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]       , y[n,:]       )
      k2 = h * fun( x[n]+1/2*h , y[n,:]+1/2*k1)
      k3 = h * fun( x[n]+3/4*h , y[n,:]+3/4*k2)
      y_ = y[n,:] + 2/9*k1 + 1/3*k2 + 4/9*k3
      k4 = h * fun( x[n]+h     , y_)
      y[n+1,:] = y[n,:] + 7/24*k1 + 1/4*k2 + 1/3*k3 + 1/8*k4
   return y

   # Van der Houwen's/Wray third-order method
def wray3( fun, x, y0 ):
   '''
    Van der Houwen's/Wray third-order method
    -----------------------
    Butcher Table:

    0    |  0     0     0     
    8/15 | 8/15   0     0     
    2/3  | 1/4  5/12    0     
    -----------------------
         | 1/4   0    3/4 
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]             , y[n,:]        )
      k2 = h * fun( x[n]+8.0*h/15.0  , y[n,:]+8.0/15.0*k1        )
      k3 = h * fun( x[n]+2.0*h/3.0  , y[n,:]+ k1/4.0 + 5.0*k2/12.0  )
      y[n+1,:] = y[n,:] + 1.0*k1/4.0 + 0.0*k2 + 3.0*k3/4.0
   return y

## HW2 Extra Credit
# Heun 2nd order
def heun( fun, x, y0):
   '''
   Explicit Heun 2nd order
   -----------------------------
    Butcher Table:

    0   | 0     0
    1   | 1     0
    -----------------
        | 1/2   1/2
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size ( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0,N-1):
      k1 = h * fun( x[n]     , y[n,:])
      k2 = h * fun( x[n]     , y[n,:]+k1)
      y[n+1,:] = y[n,:] + k1/2.0 + k2/2.0
   return y

# Kutta 3rd order
def kutta( fun, x, y0):
   '''
   Explicit Kutta 3rd order
   -----------------------------
    Butcher Table:
    0   | 0     0     0
    1/2 | 1/2   0     0
    1   |-1     2     0
    ---------------------
        | 1/6   2/3   1/6
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size ( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0,N-1):
      k1 = h * fun( x[n]     , y[n,:])
      k2 = h * fun( x[n]+h/2 , y[n,:]+k1/2)
      k3 = h * fun( x[n]+h   , y[n,:]-k1+2*k2)
      y[n+1,:] = y[n,:] + k1/6.0 + k2*2.0/3.0 + k3/6.0
   return y

# Runge-Kuta 3/8-rule 4th order
def rk38( fun, x, y0 ):
   '''
    Runge-Kutta 3/8-rule 4th order
    -----------------------------
    Butcher Table:

    0   | 0     0     0     0
    1/3 | 1/3   0     0     0
    2/3 |-1/3   1     0     0
    1   | 1    -1     1     0
    -----------------------------
        | 1/8   3/8   3/8   1/8
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]         , y[n,:]        )
      k2 = h * fun( x[n]+h/3.0   , y[n,:]+k1/3.0 )
      k3 = h * fun( x[n]+h*2/3.0 , y[n,:]-k1/3.0 + k2)
      k4 = h * fun( x[n]+h       , y[n,:]+k1 - k2)
      y[n+1,:] = y[n,:] + k1/8.0 + k2*3.0/8.0 + k3*3.0/8.0 + k4/8.0
   return y

# Ralston 4th order
def Ralston( fun, x, y0 ):
   '''
    Ralston 4th order
    -----------------------------
    Butcher Table:

    0          | 0           0           0           0  
    0.4        | 0.4         0           0           0
    0.45573725 | 0.29697761  0.15875964  0           0
    1          | 0.21810040 -3.05096516  3.83286576  0
    -----------------------------------------------------------
               | 0.17476028 -0.55148066  1.20553560  0.17118478
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]              , y[n,:] )
      k2 = h * fun( x[n]+0.4*h        , y[n,:]+0.4*k1 )
      k3 = h * fun( x[n]+0.45573725*h , y[n,:]+0.29697761*k1 + 0.15875964*k2 )
      k4 = h * fun( x[n]+h            , y[n,:]+0.21810040*k1 - 3.05096516*k2 + 3.83286576*k3 )
      y[n+1,:] = y[n,:] + 0.17476028*k1 - 0.55148066*k2 + 1.20553560*k3 + 0.17118478*k4
   return y

## HW3
# Runge-Kutta 4th for 2nd order ODE
def rk4_2nd(f, g, t, initial):
   N = np.size( t )
   h = t[1] - t[0]
   I = np.size(initial)
   xv = np.zeros((N,I))
   xv[:,0] = initial[0]
   xv[:,1] = initial[1]
   for n in range(0, N-1):
      k1 = h * f(t[n],       xv[n,0],        xv[n,1])
      l1 = h * g(t[n],       xv[n,0],        xv[n,1])
      k2 = h * f(t[n]+h/2.0, xv[n,0]+k1/2.0, xv[n,1]+l1/2.0)
      l2 = h * g(t[n]+h/2.0, xv[n,0]+k1/2.0, xv[n,1]+l1/2.0)
      k3 = h * f(t[n]+h/2.0, xv[n,0]+k2/2.0, xv[n,1]+l2/2.0)
      l3 = h * g(t[n]+h/2.0, xv[n,0]+k2/2.0, xv[n,1]+l2/2.0)
      k4 = h * f(t[n]+h    , xv[n,0]+k3    , xv[n,1]+l3    )
      l4 = h * g(t[n]+h    , xv[n,0]+k3    , xv[n,1]+l3    )
      #print(k1,l1,k2,l2,k3,l3,k4,l4)
      xv[n+1,0] = xv[n,0] + k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0
      xv[n+1,1] = xv[n,1] + l1/6.0 + l2/3.0 + l3/3.0 + l4/6.0
   return xv

# Harmonic Oscillator Leapfrog
def leapfrog_2nd(f,g,t,initial):
   N = np.size( t )
   h = t[1] - t[0]
   I = np.size(initial)
   xv = np.zeros((N,I))
   xv[:,0] = initial[0]
   xv[:,1] = initial[1]
   for n in range(0, N-1):
      xv[n+1,0] = (1-h**2/2) *xv[n,0] + (h)       *xv[n,1]
      xv[n+1,1] = (-h+h**3/4)*xv[n,0] + (1-h**2/2)*xv[n,1]
   return xv

def leapfrog_2nd_f_correction(f,g,t,initial):
   N = np.size( t )
   h = t[1] - t[0]
   I = np.size(initial)
   xv = np.zeros((N,I))
   xv[:,0] = initial[0]
   xv[:,1] = initial[1]
   Omega_new = np.sin(h/2)/(h/2)
   for n in range(0, N-1):
      xv[n+1,0] = (1-Omega_new**2*h**2/2) *xv[n,0] + (h)       *xv[n,1]
      xv[n+1,1] = (-Omega_new**2*h+Omega_new**4*h**3/4)*xv[n,0] + (1-Omega_new**2*h**2/2)*xv[n,1]
   return xv
