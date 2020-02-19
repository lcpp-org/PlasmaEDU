#function plot3D_currentloops( CurrentLoops, nSamplePoints, FigureID )
import matplotlib.pyplot as plt
import numpy as np
import config as cfg
import roto as roto
from mpl_toolkits import mplot3d


def plot3D_currentloops(nSamplePoints,FigureID):

    plt.figure( FigureID )
    ax = plt.axes(projection='3d')


    for ii in range(0 ,cfg.CurrentLoops.shape[0]):
        oclab   =   cfg.CurrentLoops[ii][0:3]
        nh      =   cfg.CurrentLoops[ii][3:6]
        I0      =   cfg.CurrentLoops[ii][6]
        Ra      =   cfg.CurrentLoops[ii][7]
        Nw      =   cfg.CurrentLoops[ii][8]

        dz  =   2.0*np.pi/nSamplePoints
        cz  =   np.cos(np.arange(0.0,dz*nSamplePoints,dz))
        sz  =   np.sin(np.arange(0.0,dz*nSamplePoints,dz))

        LoopX   =   Ra * cz
        LoopY   =   Ra * sz
        LoopZ   =  0.0 * cz

        ROT_LAB_LOOP    =   roto.ROT( nh )
        CurrentLoopX=np.zeros(sz.size)
        CurrentLoopY=np.zeros(sz.size)
        CurrentLoopZ=np.zeros(sz.size)

        for jj in range(0,sz.size):
            multiply=[ LoopX[jj], LoopY[jj], LoopZ[jj] ]
            CurrentLoopX[jj] = oclab[0] + ROT_LAB_LOOP[0].dot(multiply)
            CurrentLoopY[jj] = oclab[1] + ROT_LAB_LOOP[1].dot(multiply)
            CurrentLoopZ[jj] = oclab[2] + ROT_LAB_LOOP[2].dot(multiply)

#           % plot 3D loop center
        ax.scatter3D(  oclab[0], oclab[1], oclab[2])
        ax.hold(True)

        #% plot 3D loop coil
        ax.plot3D( CurrentLoopX, CurrentLoopY, CurrentLoopZ )
        plt.grid()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
