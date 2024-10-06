import numpy as np
import pytest
from matplotlib import pyplot as plt

from railmover.mover import Mover
from railmover.rail import BSpline


def test_mover():
    # Curve only includes one copy of endpoints for no singularities in ds/du at curve ends
    R=np.array([[-200.0, -10.0,  10.0,   0.0, -10.0,  10.0, 200.0],
                [ 200.0, -30.0,   0.0,  10.0,   0.0, -30.0, 200.0]])
    rail=BSpline(control_points=R)
    mover=Mover(rail=rail,s0=0.0,v0=0.0,g0=np.array([[0.0],[-10.0]]))
    t=0.0
    dt=1.0/128.0
    ts = []
    vs = []
    ss = []
    us = []
    while mover.u<(rail.max_u-0.5): # let the cart get 50m along the track
        ts.append(t)
        ss.append(mover.s)
        vs.append(mover.v)
        us.append(mover.u)
        mover.step(dt=dt)
        t+=dt
    ts=np.array(ts)
    vs=np.array(vs)
    ss=np.array(ss)
    us=np.array(us)
    plt.figure()
    plt.plot(ts,us)
    plt.title("t vs u")
    plt.xlabel("t/s")
    plt.ylabel("u")

    plt.figure()
    plt.plot(ts, vs)
    plt.title("t vs v")
    plt.xlabel("t/s")
    plt.ylabel("v/(m/s)")

    plt.figure()
    plt.plot(ts, ss)
    plt.title("t vs s")
    plt.xlabel("t/s")
    plt.ylabel("s/m")

    plt.figure()
    rs=rail.r(us)
    plt.plot(rs[0,:],rs[1,:])
    plt.xlabel("x/m")
    plt.ylabel("y/m")
    plt.axis('equal')

    plt.figure('Conservation of energy')
    rs = rail.r(us)
    plt.plot(vs**2,rs[1, :])
    plt.ylabel("y/m")
    plt.xlabel("ke/(m/s)**2")

    plt.show()
