#function dY=blines(s,y)
from numpy import linalg as LA
import numpy as np
import bfield as bf


def blines(y,x):
    X=y[0]
    Y=y[1]
    Z=y[2]
    direction=y[3]
    B = bf.bfield( [X,Y,Z])
    Bnorm = LA.norm(B)
    dY=np.zeros(4)
    dY[0] = direction * B[0]/Bnorm
    dY[1] = direction * B[1]/Bnorm
    dY[2] = direction * B[2]/Bnorm
    dY[3] = 0.0
    return dY
