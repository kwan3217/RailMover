import numpy as np
import pytest
from matplotlib import pyplot as plt

from railmover.rail import BSpline

@pytest.mark.parametrize(
    "R",
    [
        # Curve only includes one copy of endpoints for no singularities in ds/du at curve ends
        np.array([[-200.0,-10.0, 10.0,  0.0,-10.0, 10.0,200.0],
                  [ 200.0,-30.0,  0.0, 10.0,  0.0,-30.0,200.0],
                  [   0.0,  2.0,  4.0,  6.0,  8.0, 10.0, 12.0]]),
        #Curve includes multiple copies of endpoints so that the curve reaches those endpoints
        np.array([[-40.0,-40.0,-40.0,-10.0, 10.0,  0.0,-10.0, 10.0, 40.0, 40.0, 40.0],
                  [ 15.0, 15.0, 15.0,-30.0,  0.0, 10.0,  0.0,-30.0, 15.0, 15.0, 15.0]]),
    ]
)
def test_bspline(R:np.array):
    plt.figure('Curve')
    plt.plot(R[0,:],R[1,:],'o-')
    plt.axis('equal')
    plt.figure('First Derivative')
    plt.plot(R[0,:]*0,R[1,:]*0,'o-')
    plt.axis('equal')
    plt.figure('Second Derivative')
    plt.plot(R[0,:]*0,R[1,:]*0,'o-')
    plt.axis('equal')
    plt.figure('Third Derivative')
    plt.plot(R[0,:]*0,R[1,:]*0,'o-')
    plt.axis('equal')
    curve=BSpline(R)
    for i in range(curve.min_i,curve.max_i+1):
        u=np.linspace(i,i+1,101)
        r=curve.r(u)
        plt.figure('Curve')
        plt.plot(r[0,:],r[1,:],'-',label=f'Segment {i}')
        r = curve.dnrdun(u,1)
        plt.figure('First Derivative')
        plt.plot(r[0, :], r[1, :], '-', label=f'Segment {i}')
        r = curve.dnrdun(u, 2)
        plt.figure('Second Derivative')
        plt.plot(r[0, :], r[1, :], '-', label=f'Segment {i}')
        r = curve.dnrdun(u, 3)
        plt.figure('Third Derivative')
        plt.plot(r[0, :], r[1, :], '-', label=f'Segment {i}')
    plt.legend()
    curve.plot_frame(curve.min_u,curve.max_u,0.01,20)
    plt.show()