import numpy as np
import matplotlib.pylab as plt

def takacs_error(fd,ft,dims,M):
    # This function computes the total, dissipation, and dispersion error caused
    # by numerically approximating a solution.
    # Error estimation follows the method developed by Takacs:
    #       Lawrence Takacs. "A Two-STep Scheme for the Advectino Equation with
    #           Minimized Dissipation and Dispersion Errors." Monthly Weather
    #           Review, 113:1050-1065, February 1985.
    #
    # fd   = "discretized" (numerical) solution
    # ft   = "true" (analytical) solution
    # dims = number of dimensions - kept general for 1D or 2D/1D1V simulations
    # M    = number of total gridpoints (i.e. in 1D1V, M = Nx * Nv )

    E_tot      = sum(sum(((ft - fd))**2.))/float(M)

    mean_fd   = sum(sum(fd))/float(M)
    mean_ft   = sum(sum(ft))/float(M)
    sigma_fd  = sum(sum((fd-mean_fd)**2.0))
    sigma_ft  = sum(sum((ft-mean_ft)**2.0))
    toprow    = sum(sum((fd-mean_fd)*(ft-mean_ft)))
    botrow_d  = sum(sum((fd-mean_fd)**2.0))
    botrow_t  = sum(sum((ft-mean_ft)**2.0))
    sigma_fd  = sigma_fd/float(M)
    sigma_ft  = sigma_ft/float(M)
    sigd      = np.sqrt(sigma_fd)
    sigt      = np.sqrt(sigma_ft)

    E_diss    = ((sigt-sigd)**2.0 + (mean_ft - mean_fd)**2.0)

    ro        = toprow/np.sqrt(botrow_d*botrow_t)
    E_disp    = (2.0*(1.0-ro)*sigt*sigd)

    return(E_tot,E_diss,E_disp)


def plotting(xgrid,vgrid,x,v,f,scalar_field,f0,timestep,plot_save,plot_show):
    if (plot_show == 1):
        plt.clf()
        plt.ion()
    # plt.subplot(2,1,1)
    plt.pcolormesh(xgrid,vgrid,f)
    plt.axis([x[0],x[-1],v[0],v[-1]])
    plt.xlabel('X',fontsize=20)
    plt.ylabel('V',fontsize=20)
    plt.title('Phase Space')

    # Electric Field Comparison
    # time = 0.02*timestep
    # Et = 4.0*0.01 * 0.3677 * np.exp(-0.1533*time)*np.sin(0.5*x)*np.cos(1.4156*time - 0.5326245)
    # plot_title = 't = '+str(timestep*0.02)
    # plt.plot(x,scalar_field)
    # plt.plot(x,Et)
    # plt.title(plot_title)

    # plt.subplot(2,1,2)
    # plt.plot(x,potential)
    # plt.xlabel('X')
    # plt.ylabel('$\phi$')
    if (plot_show == 1):
        plt.draw()
        plt.pause(.05)

    if (plot_save == 1):
        FIG_NAME = 'OUT/f_nonlinear_' + str(timestep) + '.png'
        plt.savefig(FIG_NAME)
