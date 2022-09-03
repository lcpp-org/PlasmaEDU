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

# Kutta 3rd order
def k3( fun, x, y0 ):
   '''
    Kutta 3rd order
    -----------------------
    Butcher Table:

    0   | 0     0     0     
    1/2 | 1/2   0     0     
    1   | -1    2     0     
    -----------------------
        | 1/6   2/3   1/6  
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]       , y[n,:]        )
      k2 = h * fun( x[n]+h/2.0 , y[n,:]+k1/2.0 )
      k3 = h * fun( x[n]-h     , y[n,:]-k1+k2/2.0 )
      y[n+1,:] = y[n,:] + k1/6.0 + 2*k2/3.0 + k3/6.0
   return y

   # Heun 3rd order
def heun3( fun, x, y0 ):
   '''
    Heun 3rd order
    -----------------------
    Butcher Table:

    0   | 0     0     0     
    1/3 | 1/3   0     0     
    2/3 | 0    2/3    0     
    -----------------------
        | 1/4   0    3/4 
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]       , y[n,:]        )
      k2 = h * fun( x[n]+h/3.0 , y[n,:]+k1/3.0 )
      k3 = h * fun( x[n]+2*h/3.0     , y[n,:]+2*k2/3.0 )
      y[n+1,:] = y[n,:] + k1/4.0 + 3*k3/4.0
   return y

   # Ralston 3rd order
def ralston3( fun, x, y0 ):
   '''
    Ralston 3rd order
    -----------------------
    Butcher Table:

    0   | 0     0     0     
    1/2 | 1/2   0     0     
    3/4 | 0    3/4    0     
    -----------------------
        | 2/9   1/3    4/9 
   '''
   N = np.size( x )
   h = x[1] - x[0]
   I = np.size( y0 )
   y = np.zeros((N,I))
   y[0,:] = y0
   for n in range(0, N-1):
      k1 = h * fun( x[n]       , y[n,:]        )
      k2 = h * fun( x[n]+h/2.0 , y[n,:]+k1/2.0 )
      k3 = h * fun( x[n]+3*h/4.0     , y[n,:]+3*k2/4.0 )
      y[n+1,:] = y[n,:] + 2*k1/9.0 + k2/3 + 4*k3/9.0
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
