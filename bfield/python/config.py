import numpy as np
#Function that stores all global variables and constants


#constants


#Global Variables
#% CurrentLoops = [  Xc,  Yc,  Zc,  nx,  ny,  nz,  I0,   Ra,   N_windings; ...
CurrentLoops = np.array([[ -0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 10e3, 0.025, 1],
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 10e3, 0.05,  1],
                        [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 10e3, 0.10,  1],
                        [0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 10e3, 0.05,  1],
                        [0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 10e3, 0.025, 1]])
