function ROT = roto( Nhat )

Nx    = Nhat(1);
Ny    = Nhat(2);
Nz    = Nhat(3);

Nxy   = sqrt( Nx*Nx + Ny*Ny );
Nnorm = sqrt( Nx*Nx + Ny*Ny + Nz*Nz );

cth   = Nx/Nxy;
sth   = Ny/Nxy;

cphi  = Nxy/Nnorm;
sphi  = Nz/Nnorm;

R0 = [ 0 0 1; 1 0 0; 0 1 0];
R1 = [cphi 0 sphi; 0 1 0; -sphi 0 cphi];
R2 = [cth -sth 0; sth cth 0; 0 0 1];

ROT = R0 * R1 * R2;
