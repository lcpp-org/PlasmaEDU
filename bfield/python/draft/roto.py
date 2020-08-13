#function ROT = roto( Nhat )
import numpy as np

def ROT(Nhat):

    Nx    = Nhat[0]
    Ny    = Nhat[1]
    Nz    = Nhat[2]

    Nxy   = np.sqrt( Nx*Nx + Ny*Ny );
    Nnorm = np.sqrt( Nx*Nx + Ny*Ny + Nz*Nz );

    cth   = Nx/Nxy;
    sth   = Ny/Nxy;

    cphi  = Nxy/Nnorm;
    sphi  = Nz/Nnorm;


    R0 = np.array([[0,0,1],[1,0,0],[0,1,0]])
    R1 = np.array([[cphi,0,sphi],[0,1,0],[-sphi,0,cphi]])
    R2 = np.array([[cth,-sth,0],[sth,cth,0],[0,0,1]])

    ROT = (R0.dot(R1)).dot(R2)
    return ROT
