"""
Describe purpose of this script here

Created: 10/3/24
"""
import numpy as np
import pytest
from kwanmath.vector import vnormalize
from matplotlib import pyplot as plt

from railmover.rail import BSpline, CircularRail, HelicalRail


@pytest.mark.parametrize(
    "u,n_r,ref_i",
    [(0.0,10,{0}),
     (np.array([0]),10,{0}),
     (np.array([0.5,1.5,2.5]),10,{0,1,2}),
     (np.array([9.5,10.0,10.5]),10,{6}),
     ]
)
def test_constrain_i(u,n_r,ref_i):
    curve=BSpline(np.zeros((2,n_r)))
    assert np.all(curve.select_i(u)==ref_i)


def test_circular_rail_position():
    circle = CircularRail(radius=5)
    angle = np.pi / 4  # 45 degrees
    position = circle.r(angle)
    assert np.isclose(np.linalg.norm(position), 5, rtol=1e-5)

def test_circular_rail_dsdu():
    circle = CircularRail(radius=5)
    angle = np.pi / 4
    dsdu = circle.dsdu(angle)
    assert np.isclose(dsdu, 5, rtol=1e-5)

def test_helical_rail_position():
    helix = HelicalRail()
    u = 2 * np.pi
    position = helix.r(u)
    assert np.isclose(np.linalg.norm(position[:2]), 1, rtol=1e-5)
    assert np.isclose(position[2], 1, rtol=1e-5)

def test_circular_rail_curvature():
    # Curvature should be constant and equal to 1/radius
    circle = CircularRail(radius=5)
    radius = circle.radius
    for u in np.linspace(0, 2 * np.pi, 100):
        curvature = circle.kappa(u)
        assert np.isclose(curvature, 1 / radius, rtol=1e-5)

def test_circular_rail_torsion():
    # Torsion should be zero for a circle
    circle = CircularRail(radius=5)
    for u in np.linspace(0, 2 * np.pi, 100):
        torsion = circle.tau(u)
        assert np.isclose(torsion, 0, rtol=1e-5)

def test_helical_rail_curvature():
    # Curvature should be constant
    helix = HelicalRail()
    # curvature formula for a helix from eqn (4) at https://mathworld.wolfram.com/Helix.html
    c = helix.pitch/(2*np.pi)
    r = 1  # Assuming a unit circle
    expected_curvature = r/(r**2+c**2)
    for u in [0.0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]:
        curvature = helix.kappa(u)
        assert np.isclose(curvature, expected_curvature, rtol=1e-5)

def test_helical_rail_torsion():
    # Torsion should be constant
    helix = HelicalRail()
    c = helix.pitch/(2*np.pi)
    r = 1  # Assuming a unit circle
    # torsion formula for a helix from ibid eqn(6)
    expected_torsion = c / (r**2+c**2)
    for u in [0.0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]:
        torsion = helix.tau(u)
        assert np.isclose(torsion, expected_torsion, rtol=1e-5)

@pytest.mark.parametrize(
    'u,ref_T',
    [(   0.0   ,np.array([[ 0.0],[ 1.0],[0.0]])),
     (  np.pi/2,np.array([[-1.0],[ 0.0],[0.0]])),
     (  np.pi  ,np.array([[ 0.0],[-1.0],[0.0]])),
     (3*np.pi/2,np.array([[ 1.0],[ 0.0],[0.0]])),
     (2*np.pi  ,np.array([[ 0.0],[ 1.0],[0.0]])),
     ]
)
def test_circle_T(u,ref_T):
    circle = CircularRail(radius=5)
    calc_T=circle.T(u)
    assert np.allclose(calc_T,ref_T)


@pytest.mark.parametrize(
    'u,ref_N',
    [(   0.0   ,np.array([[-1.0],[ 0.0],[0.0]])),
     (  np.pi/2,np.array([[ 0.0],[-1.0],[0.0]])),
     (  np.pi  ,np.array([[ 1.0],[ 0.0],[0.0]])),
     (3*np.pi/2,np.array([[ 0.0],[ 1.0],[0.0]])),
     (2*np.pi  ,np.array([[-1.0],[ 0.0],[0.0]])),
     ]
)
def test_circle_N(u,ref_N):
    circle = CircularRail(radius=5)
    calc_N=circle.N(u)
    assert np.allclose(calc_N,ref_N)


@pytest.mark.parametrize(
    'u,ref_B',
    [(   0.0   ,np.array([[ 0.0],[ 0.0],[1.0]])),
     (  np.pi/2,np.array([[ 0.0],[ 0.0],[1.0]])),
     (  np.pi  ,np.array([[ 0.0],[ 0.0],[1.0]])),
     (3*np.pi/2,np.array([[ 0.0],[ 0.0],[1.0]])),
     (2*np.pi  ,np.array([[ 0.0],[ 0.0],[1.0]])),
     ]
)
def test_circle_B(u,ref_B):
    circle = CircularRail(radius=5)
    calc_B=circle.B(u)
    assert np.allclose(calc_B,ref_B)

@pytest.mark.parametrize(
    'rail,u0,u1,du,step',
    [(CircularRail(radius=5),0,2*np.pi,0.01,10),
     (HelicalRail(radius=4,pitch=1),0,20*np.pi,0.01,100),
     ]
)
def test_frame(rail,u0,u1,du,step):
    rail.plot_frame(u0,u1,du,step)
    plt.show()


@pytest.mark.parametrize(
    'u',
    [   0.0   ,
       np.pi/2,
       np.pi  ,
     3*np.pi/2,
     2*np.pi  ,
     ]
)
def test_helix_T(u):
    rail = HelicalRail(radius=5,pitch=1)
    ref_T=vnormalize(np.array([[-np.sin(u)],[np.cos(u)],[rail.pitch/(rail.radius*2*np.pi)]]))
    calc_T=rail.T(u)
    assert np.allclose(calc_T,ref_T)


@pytest.mark.parametrize(
    'u',
    [  np.pi/2,
        0.0   ,
       np.pi  ,
     3*np.pi/2,
     2*np.pi  ,
     ]
)
def test_helix_N(u):
    rail = HelicalRail(radius=5,pitch=1)
    ref_N=vnormalize(np.array([[-np.cos(u)],[-np.sin(u)],[0.0]]))
    calc_N=rail.N(u)
    assert np.allclose(calc_N,ref_N)

