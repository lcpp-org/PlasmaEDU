import numpy as np

def psi_iter_like(R,Z):

        R0   =   6.2
        A    =  -0.155
        Psi0 = 202.92

        # Normalized coordinates w.r.t. major radius
        x = R/R0
        y = Z/R0

        # Powers of x and y and log
        x2 = x*x
        x4 = x2*x2
        y2 = y*y
        y4 = y2*y2
        lnx = np.log(x)

        # Single-null Grad Shafranov functions

        psi_i = np.zeros(12)
        coeff = np.zeros(12)

        psi_i[0] = 1.0
        psi_i[1] = x2
        psi_i[2] = y2 - x2*lnx
        psi_i[3] = x4 - 4.0*x2*y2
        psi_i[4] = 2.0*y4 - 9.0*y2*x2 + 3.0*x4*lnx - 12.0*x2*y2*lnx
        psi_i[5] = x4*x2 - 12.0*x4*y2 + 8.0*x2*y4
        psi_i[6] = 8.0*y4*y2 - 140.0*y4*x2 +75.0*y2*x4 -15.0*x4*x2*lnx + 180.0*x4*y2*lnx - 120.0*x2*y4*lnx
        psi_i[7] = y
        psi_i[8] = y*x2
        psi_i[9] = y*y2 - 3.0*y*x2*lnx
        psi_i[10] = 3.0*y*x4 - 4.0*x2*y2*y
        psi_i[11] = 8.0*y4*y - 45.0*y*x4 - 80.0*y2*y*x2*lnx + 60.0*y*x4*lnx

        # Coefficients for ITER-like magnetic equilibrium
        coeff[0] =  1.00687012e-1
        coeff[1] =  4.16274456e-1
        coeff[2] = -6.53880989e-1
        coeff[3] = -2.95392244e-1
        coeff[4] =  4.40037966e-1
        coeff[5] = -4.01807386e-1
        coeff[6] = -1.66351381e-2
        coeff[7] =  1.92944621e-1
        coeff[8] =  8.36039453e-1
        coeff[9] = -5.30670714e-1
        coeff[10]= -1.26671504e-1
        coeff[11]=  1.47140977e-2

        psi = np.dot(coeff, psi_i) + x4/8.0 + A * (0.5*x2*lnx - x4/8.0)

        Psi = Psi0*psi

        return Psi
