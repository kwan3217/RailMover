import numpy as np
import pytest
from matplotlib import pyplot as plt

from railmover.mover import Cart
from railmover.rail import BSpline, CircularRail, StraightRail, LinearSpline


def test_mover():
    # Curve only includes one copy of endpoints for no singularities in ds/du at curve ends
    R=np.array([[-200.0, -10.0,  10.0,   0.0, -10.0,  10.0, 200.0],
                [ 200.0, -30.0,   0.0,  10.0,   0.0, -30.0, 200.0]])
    rail=BSpline(control_points=R)
    mover=Cart(rail=rail, s0=0.0, dsdt0=0.0, g=np.array([[0.0], [-10.0]]))
    t=0.0
    dt=1.0/128.0
    ts = []
    vs = []
    ss = []
    us = []
    while mover.u<(rail.max_u-0.5): # let the cart get 50m along the track
        ts.append(t)
        ss.append(mover.s)
        vs.append(mover.dsdt)
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


def test_accelerometer_centrip():
    """
    Test the accelerometer on a circular rail with no gravity. Only
    acceleration should be centripetal.
    :return: None, but raises an exception if test fails
    """
    rail=CircularRail(radius=1)
    mover=Cart(rail, u0=0.0, dsdt0=1.0, g=np.array([[0.0], [0.0], [0.0]]), crr=0, kd=0)
    # expected centripetal force=v**2/r=1**2/1=1, in direction towards center
    # Since rail is horizontal and centered on origin, initial centripetal
    # acceleration is -\hat{i}
    ref_a_nongrav=np.array([[-1.0],[0.0],[0.0]])
    ref_asT=0.0 # No tangential force
    asT,a_nongrav=mover.a()
    assert np.isclose(asT,ref_asT)
    assert np.allclose(a_nongrav,ref_a_nongrav)


def test_accelerometer_grav():
    """
    Test the accelerometer on a straight rail and level with gravity. Accelerometer
    should feel an upward acceleration equal and opposite to gravity.
    :return: None, but raises an exception if test fails
    """
    rail=StraightRail(r0=np.array([[0.0],[0.0],[0.0]]),r1=np.array([[1.0],[0.0],[0.0]]))
    mover=Cart(rail, u0=0.0, dsdt0=1.0, g=np.array([[0.0], [0.0], [-1.0]]), crr=0, kd=0)
    # expected centripetal force=v**2/r=1**2/1=1, in direction towards center
    # Since rail is horizontal and centered on origin, initial centripetal
    # acceleration is -\hat{i}
    ref_a_nongrav=np.array([[0.0],[0.0],[1.0]])
    ref_asT=0.0 # No tangential force
    asT,a_nongrav=mover.a()
    assert np.isclose(asT,ref_asT)
    assert np.allclose(a_nongrav,ref_a_nongrav)


def test_accelerometer_grav2():
    """
    Test the accelerometer on a linear spline with one segment --
    should perfectly match test_accelerometer_grav().
    :return: None, but raises an exception if test fails
    """
    R=np.array([[ 0.0,   1.0],
                [ 0.0,   0.0],
                [ 0.0,   0.0]])
    rail=LinearSpline(control_points=R)
    mover=Cart(rail, u0=0.0, dsdt0=1.0, g=np.array([[0.0], [0.0], [-1.0]]), crr=0, kd=0)
    # expected centripetal force=v**2/r=1**2/1=1, in direction towards center
    # Since rail is horizontal and centered on origin, initial centripetal
    # acceleration is -\hat{i}
    ref_a_nongrav=np.array([[0.0],[0.0],[1.0]])
    ref_asT=0.0 # No tangential force
    asT,a_nongrav=mover.a()
    assert np.isclose(asT,ref_asT)
    assert np.allclose(a_nongrav,ref_a_nongrav)


def test_accelerometer_linearspline():
    """
    Test the accelerometer on a linear spline with one segment --
    should perfectly match test_accelerometer_grav().
    :return: None, but raises an exception if test fails
    """
    R=np.array([[ 0.0,   1.0,  2.0],
                [ 0.0,   0.0,  0.0],
                [ 1.0,   0.0,  1.0]])
    rail=LinearSpline(control_points=R)
    mover=Cart(rail, u0=0.5, dsdt0=1.0, g=np.array([[0.0], [0.0], [-1.0]]), crr=0, kd=0)
    # expected centripetal force=v**2/r=1**2/1=1, in direction towards center
    # Since rail is horizontal and centered on origin, initial centripetal
    # acceleration is -\hat{i}
    ref_a_nongrav=np.array([[0.0],[0.0],[1.0]])
    ref_asT=0.0 # No tangential force
    asT,a_nongrav=mover.a()
    mover.u=1.5
    asT,a_nongrav=mover.a()
    assert np.isclose(asT,ref_asT)
    assert np.allclose(a_nongrav,ref_a_nongrav)
