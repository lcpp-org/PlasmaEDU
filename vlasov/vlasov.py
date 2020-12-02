'''

    VLASOV-POISSON PROBLEM IN 1D1V

    Shane Keniley, Davide Curreli

    4/15/2017

    '''

############################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import os
from inputdata import *
from data_analysis import plotting
#from pylab import plot, axis, show

def boundary(f,b1,b2,dim):
    # Periodic Boundary Conditions in space
    # Open boundaries in velocity
    if (dim==0):
        # Boundary Conditions - left
        f[b1-1,:] = np.copy(f[b2-0,:])
        f[b1-2,:] = np.copy(f[b2-1,:])
        f[b1-3,:] = np.copy(f[b2-2,:])

        # Boundary Conditions - right
        f[b2+1,:] = np.copy(f[b1+0,:])
        f[b2+2,:] = np.copy(f[b1+1,:])
        f[b2+3,:] = np.copy(f[b1+2,:])

    if (dim==1):
        # Boundary Conditions - bottom
        f[:,b1-1] = 0.0
        f[:,b1-2] = 0.0
        f[:,b1-3] = 0.0

        # Boundary Conditions - top
        f[:,b2+1] = 0.0
        f[:,b2+2] = 0.0
        f[:,b2+3] = 0.0


def advect_space(f,c,I1,I2,J1,J2,output,scheme):
    cl = len(c)/2
    vl = J1
    vc = J2/2 + 2
    vr = J2 + 1

    # if (scheme==)
    for i in range(I1,I2+1):
        if (scheme==4):
        # Fourth Order Upwind
            output[i,vl:vc] = -(1.0/12.0)*c[:cl]*(1.0*f[i+3,vl:vc] - 6.0*f[i+2,vl:vc] + 18.0*f[i+1,vl:vc] - 10.0*f[i,vl:vc] - 3.0*f[i-1,vl:vc])
            output[i,vc:vr] = -(1.0/12.0)*c[cl:]*(3.0*f[i+1,vc:vr] + 10.0*f[i,vc:vr] - 18.0*f[i-1,vc:vr] + 6.0*f[i-2,vc:vr] - 1.0*f[i-3,vc:vr])

        # Third Order Upwind
        elif (scheme==3):
            output[i,vl:vc]     = - (1.0/6.0)*c[:cl]*(-1.0*f[i+2,vl:vc] + \
            6.0*f[i+1,vl:vc] - 3.0*f[i,vl:vc] - 2.0*f[i-1,vl:vc] )

            output[i,vc:vr] = - (1.0/6.0)*c[cl:] * (2.0*f[i+1,vc:vr] + \
            3.0*f[i,vc:vr] - 6.0*f[i-1,vc:vr] + f[i-2,vc:vr] )

        elif (scheme==2):
        # Second order Upwind
            output[i,vl:vc] = -(1.0/2.0)*c[:cl]*(-1.0*f[i+2,vl:vc] + 4.0*f[i+1,vl:vc] - 3.0*f[i,vl:vc])
            output[i,vc:vr] = -(1.0/2.0)*c[cl:]*(3.0*f[i,vc:vr] - 4.0*f[i-1,vc:vr] + 1.0*f[i-2,vc:vr])

        # First Order Upwind
        # output[i,vl:vc] = -c[:cl] * ( f[i+1,vl:vc] - f[i,vl:vc] )
        # output[i,vc:vr] = -c[cl:] * ( f[i,vc:vr] - f[i-1,vc:vr] )

    return output

def advect_velocity(f,c,I1,I2,J1,J2,output,scheme):
    cl = len(c)/2
    vl = I1
    vc = I2/2 + 2
    vr = I2+1

    negarray = np.where(c < 0)[0]
    posarray = np.where(c > 0)[0]

    for j in range(J1,J2+1):

        if (scheme==4):
        # Fourth Order Upwind
            output[negarray+I1,j] = -(1.0/12.0)*c[negarray]*(1.0*f[negarray+I1,j+3] - 6.0*f[negarray+I1,j+2] + 18.0*f[negarray+I1,j+1] - 10.0*f[negarray+I1,j] - 3.0*f[negarray+I1,j-1])
            output[posarray+I1,j] = -(1.0/12.0)*c[posarray]*(3.0*f[posarray+I1,j+1] + 10.0*f[posarray+I1,j] - 18.0*f[posarray+I1,j-1] + 6.0*f[posarray+I1,j-2] - 1.0*f[posarray+I1,j-3])
        elif (scheme==3):
        # Third Order Upwind
            output[negarray+I1,j] = -(1.0/6.0)*c[negarray]*(-1.0*f[negarray+I1,j+2] + 6.0*f[negarray+I1,j+1] - 3.0*f[negarray+I1,j] - 2.0*f[negarray+I1,j-1])
            output[posarray+I1,j] = -(1.0/6.0)*c[posarray]*( 2.0*f[posarray+I1,j+1] + 3.0*f[posarray+I1,j] - 6.0*f[posarray+I1,j-1] + 1.0*f[posarray+I1,j-2])
        elif (scheme==2):
        # Second Order Upwind
            output[negarray+I1,j] = -(1.0/2.0)*c[negarray]*(-1.0*f[negarray+I1,j+2] + 4.0*f[negarray+I1,j+1] - 3.0*f[negarray+I1,j])
            output[posarray+I1,j]   = -(1.0/2.0)*c[posarray]*( 3.0*f[posarray+I1,j]   - 4.0*f[posarray+I1,j-1] + 1.0*f[posarray+I1,j-2])


    return output

def rk4(f,courant,I1,I2,J1,J2,h,dimension,scheme):
    output_size = np.shape(f)
    output = np.zeros(shape=output_size)
    courant = courant/h
    if (dimension==0):

        k1 = advect_space(f,courant,I1,I2,J1,J2,output,scheme)
        boundary(k1,I1,I2,dimension)
        output = output*0.0

        k2 = advect_space(f+k1*h/2.0,courant,I1,I2,J1,J2,output,scheme)
        boundary(k2,I1,I2,dimension)
        output = output*0.0

        k3 = advect_space(f+k2*h/2.0,courant,I1,I2,J1,J2,output,scheme)
        boundary(k3,I1,I2,dimension)
        output = output*0.0

        k4 = advect_space(f+k3*h,courant,I1,I2,J1,J2,output,scheme)
        boundary(k4,I1,I2,dimension)

    else:
        k1 = advect_velocity(f,courant,I1,I2,J1,J2,output,scheme)
        boundary(k1,J1,J2,dimension)
        output = output*0.0

        k2 = advect_velocity(f+k1*h/2.0,courant,I1,I2,J1,J2,output,scheme)
        boundary(k2,J1,J2,dimension)
        output = output*0.0

        k3 = advect_velocity(f+k2*h/2.0,courant,I1,I2,J1,J2,output,scheme)
        boundary(k3,J1,J2,dimension)
        output = output*0.0

        k4 = advect_velocity(f+k3*h,courant,I1,I2,J1,J2,output,scheme)
        boundary(k4,J1,J2,dimension)

    yfinal = f + h*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    return (yfinal)


# Apply boundaries to initial condition
f_initial = np.copy(f1)
boundary(f1,I1,I2,0)
boundary(f1,J1,J2,1)

# f1_0 = f1

# Laplacian matrix for Poisson solver
A = np.zeros((Nx,Nx))
for i in range (1,Nx-1):
   A[i,i-1] =  1.0
   A[i,i  ] = -2.0
   A[i,i+1] =  1.0

A[0,0] = 1.0
A[-1,-1] = 1.0

# Pre-allocation
s       =  np.zeros((Nx,1))
E       =  np.zeros((Nx,1))
phi     =  np.zeros((Nx,1))
Esq     =  np.zeros((Nx,1))
entropy =  np.zeros((Nt,1))
Enorm   =  np.zeros((Nt,1))
rho_n   =  np.zeros((Nt,1))
rho1    =  np.zeros((Nx,1))
E       = np.squeeze(E,1)
test = np.zeros((Nx,1))


# Make sure output folder exists
os.system('mkdir OUT')

Courant = V*(dt/2.0)/dX
CourantA = 0.0

# Theoretical recurrence time for linear Landau damping
t_recur = 2.0*np.pi / k / dV
# print 'Recurrence Time = '+str(t_recur)
# stop
# Start time cycle
for n in range(0,Nt):

    f1s = f1

    # Plotting Routine (if plotting is turned on - see 'input_params.py')
    if (n % nplot == 0):
        print n
        if (plot_on == 1):
            plotting(XX,VV,X,V/Vth1,f1[I1:I2+1,J1:J2+1],E,f_initial[I1:I2+1,J1:J2+1],n,plot_save,plot_show)


    # Vlasov >> Step #1, advect of dt/2 in space
    f1[:,:] = rk4(f1,Courant,I1,I2,J1,J2,dt/2.0,0,numerical_scheme)

    # >> Step #2, solve for electric field via Poisson equation
    rho_background = -Q1*np.trapz(np.trapz(f1[I1:I2+1,J1:J2+1],x=V),x=X)/L
    rho_n[n] = rho_background  # Storing background density over time for reference

    # Source vector s(x)
    for i in range(I1,I2+1):
        rho1[i-I1]  = Q1 * np.trapz(f1[i,J1:J2+1],x=V)
        s[i-I1]  = - ( rho1[i-I1] + rho_background)/eps0*dX*dX

    phi = np.linalg.solve(A,s)

    for i in range(1,Nx-1):
        E[i] = - ( phi[i+1] - phi[i-1] ) / 2.0 / dX
    E[0]   = - ( 3*phi[0]   - 4*phi[1]     + phi[2]     ) / 2.0 / dX
    E[-1]  =   ( 3*phi[-1]  - 4*phi[-2]    + phi[-3]    ) / 2.0 / dX
    # E[0] = 0.0
    # E[-1] = 0.0

    # Theoretical electric field
    # E = 4.0*0.01 * 0.3677 * np.exp(-0.1533*(n*dt))*np.sin(0.5*X)*np.cos(1.4156*(n*dt) - 0.5326245)


    # Vlasov >> Step #3, advect of dt in velocity, and dt/2 in space
    a1 = Q1/M1*E
    CourantA = a1*dt/dV

    f1[:,:] = rk4(f1,CourantA,I1,I2,J1,J2,dt   ,1,numerical_scheme)

    # Courant = V*(dt/2.0)/dX
    f1[:,:] = rk4(f1,Courant,I1,I2,J1,J2,dt/2.0,0,numerical_scheme)


    Enorm[n] = 0.5*np.trapz((E*E),x=X)

    if (np.max(Courant) >= 1.0 and cfl_space_warning == 0):
        print 'WARNING: Spatial CFL factor = '+str(np.max(Courant))+'; may be unstable!'
        cfl_space_warning = 1

    if (np.max(CourantA) >= 1.0 and cfl_velocity_warning == 0):
        print 'WARNING: Velocity CFL factor = '+str(np.max(CourantA))+'; may be unstable!'
        cfl_velocity_warning = 1




plt.close()


np.save('Enorm_linear_1',Enorm)
