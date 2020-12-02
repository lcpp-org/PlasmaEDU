import numpy as np
import matplotlib.pylab as plt
import math
from input_params import *
from scipy.optimize import curve_fit

# Variables stored from imput_params

Enorm = np.load('Enorm_linear_1.npy')
Enorm = np.sqrt(Enorm)

### RECURRENCE TIME
tr = 2.0*np.pi / 0.5 / dV
tr2 = tr / 1.5
tr3 = tr / 2.0
print tr

xtr = np.zeros((100)) + tr
ytr = np.linspace(np.min(Enorm),np.max(Enorm),100)

xtr2 = np.zeros((100)) + tr2
ytr2 = np.linspace(np.min(Enorm),np.max(Enorm),100)

xtr3 = np.zeros((100)) + tr3
ytr3 = np.linspace(np.min(Enorm),np.max(Enorm),100)

### THEORETICAL ELECTRIC FIELD
Et = np.zeros((Nx,len(time_vector)))
Et_norm = np.zeros((len(time_vector)))
for i in range(Nx):
    Et[i,:] = 4.0*0.001 * 0.3677 * np.exp(-0.1533*time_vector)*np.sin(0.5*X[i])*np.cos(1.4156*time_vector - 0.5326245)

for n in range(len(time_vector)):
    Et_norm[n] = np.sqrt(np.trapz(Et[:,n]*Et[:,n],x=X))

### DAMPING RATE FITTING
# Take derivative of Enorm
dedx = np.zeros((len(Enorm),1))
for i in range(1,len(Enorm)-1):
    dedx[i] = -(Enorm[i+1] - Enorm[i-1]) / 2.0 / dt
dedx[0] = ( 3*Enorm[0]   - 4*Enorm[1]     + Enorm[2]     ) / 2.0 / dt
dedx[-1]  = - ( 3*Enorm[-1]  - 4*Enorm[-2]    + Enorm[-3]    ) / 2.0 / dt

# Find location of zeros (only the local maxima, not minima)
dedx_temp = np.zeros((len(Enorm),1))
for i in range(1,len(dedx)-1):
    if (dedx[i]*dedx[i-1] < 0 and dedx[i-1]<0):
        dedx_temp[i] = i
dedx_0 = dedx_temp[dedx_temp != 0]



# Find values of maxima for least-squares fitting
x_max = np.zeros((len(dedx_0)))
Enorm_max = np.zeros((len(dedx_0)))
for i in range(len(dedx_0)):
    # x values (time 'locations')
    x_max[i] = time_vector[dedx_0[i]]

    # y values (maxima)
    Enorm_max[i] = Enorm[dedx_0[i]]


# Theoretical Damping Rate (for linear Landau damping)
imw = np.sqrt(np.pi/8.0) * (1.0/(k**3.0)) * np.exp(-1.0/(2.0*(k**2.0)) - (3.0/2.0))
print 'Theoretical damping rate = '+str(imw)


### LEAST SQUARES FIT TO MAXIMA :  y = a*exp(-c*t)
def func(x, a, c):
    return (a*np.exp(-c*x))


popt, pcov = curve_fit(func, x_max[1:10], Enorm_max[1:10])
print 'Numerical damping rate = '+str(popt)


xx = np.linspace(x_max[0],x_max[-1],100)
# xx = np.linspace(time_vector[1000],time_vector[3000],100)
yy = func(xx, *popt)



plt.figure(figsize=(12,10))
# plt.semilogy(time_vector,Enorm3,linewidth=2,label='Nv = 64')

plt.semilogy(time_vector,Enorm,linewidth=2)
# plt.plot(time_vector,yy,'--',linewidth=4,color='gray')
# plt.scatter(x_max,Enorm_max,c='red',s=100)
plt.ylabel('L2 Norm, Electric Field')
plt.xlabel('Time',fontsize=20)
# plt.legend()
# plt.axis([time_vector[0],time_vector[-1],np.min(Enorm),np.max(Enorm)*1.2])
plt.show()
# plt.savefig('enorm_linear_2.png')
# plt.close()
