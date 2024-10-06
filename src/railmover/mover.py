"""
Describe purpose of this script here

Created: 10/4/24
"""
import numpy as np
from kwanmath.vector import vdot

from railmover.rail import Rail


def vproj(a:np.array,b:np.array)->np.array:
    """
    Calculate the projection of b onto a. This is a vector
    with the direction of a but magnitude b cos(theta)
    where theta is the angle between the vectors.
    :param a: projection base
    :param b: vector to project
    :return: a vector in the same direction as a
    """
    return a*vdot(a,b)/vdot(a,a)


def vperp(a:np.array,b:np.array)->np.array:
    """
    Calculate the projection of b perpendicular to a. This is a vector
    with the direction perpendicular to a and magnitude b sin(theta)
    where theta is the angle between the vectors. The result is in the
    same plane as a and b, and on the same side of the plane relative to a
    as b.
    :param a: projection base
    :param b: vector to project
    :return: a vector perpendicular to a
    """
    # Calculate this by subtracting the projection from the original vector
    return b-vproj(a,b)


class Mover:
    def __init__(self,rail:Rail,s0:float,v0:float,g0:np.array,mcrr:float=0.0,cd:float=0.0):
        """
        Construct the mover
        :param rail: Rail that the mover will move along
        """
        self.rail=rail
        self.s=s0 # scalar position along track in [m]
        self.u=self.rail.calc_u(self.s) # rail parameter corresponding to this distance
        self.v=v0 # scalar speed along track in [m/s]
        self.g0=g0 # gravity vector in [m/s**2]
        self.mcrr=mcrr # mass of mover times coefficient of rolling resistance [kg]
    def step(self,dt:float)->np.array:
        # Figure out current position of cart
        r=self.rail.r(self.u)
        # unit vector in direction of track
        T=self.rail.T(self.u)
        v=T*self.v
        # Figure out forces on cart
        a=v*0.0
        # gravity
        a+=self.g0
        # rolling resistance. This is modeled as a force
        # with direction opposite the velocity and magnitude
        # proportional to the force exerted on the track. Since
        # the cart rolls, it doesn't exert any force along the
        # track, so the force exerted on the track is purely
        # perpendicular to it.
        #
        # Frr=-(v/|v|)*crr*W
        # marr=-(v/|v|)*crr*W
        # arr=-(v/|v|)*crr/m*W
        # arr=-(v/|v|)*crr/m*aW*m
        # arr=-(v/|v|)*crr*aW
        #
        # so the acceleration is independent of the mass. This doesn't
        # seem to take into account things like sticky grease. I would
        # say that crr is a weak function of mass
        gperp=vperp(T,self.g0)
        a+=gperp
        # air resistance

        # track boost/brake

        # Now what component of acceleration is actually along the direction of travel?
        # The rest is exerted on the track, and the track reacts by pushing back (3rd law)
        # and canceling it out.
        a=vdot(a,T)
        dv=a*dt
        self.v+=dv
        ds=self.v*dt
        self.s+=ds
        # du=ds*(du/ds)
        #   =ds/(ds/du)
        du=ds/self.rail.dsdu(self.u)
        self.u+=du



