function dY=blines(s,y)
global CurrentLoops
X=y(1);
Y=y(2);
Z=y(3);
direction=y(4);
B = bfield( [X;Y;Z], CurrentLoops );
Bnorm = norm(B);
dY(1) = direction * B(1)/Bnorm;
dY(2) = direction * B(2)/Bnorm;
dY(3) = direction * B(3)/Bnorm;
dY(4) = 0.0;
dY=dY';