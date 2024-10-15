"""
Describe purpose of this script here

Created: 10/4/24
"""
import numpy as np
from kwanmath.vector import vdot, vlength, vnormalize, vangle
from matplotlib import pyplot as plt

from railmover.rail import Rail


def vproj(a:np.array,b:np.array)->np.array:
    """
    Calculate the projection of b onto a. This is a vector
    with the direction of a but magnitude b cos(theta)
    where theta is the angle between the vectors.
    :param a: projection base
    :param b: vector to project
    :return: a vector in the same direction as a with length |b|cos(theta)
    """
    return vnormalize(a)*vdot(vnormalize(a),b)


def vperp(a:np.array,b:np.array)->np.array:
    """
    Calculate the projection of b perpendicular to a. This is a vector
    with the direction perpendicular to a and magnitude b sin(theta)
    where theta is the angle between the vectors. The result is in the
    same plane as a and b, and on the same side of the plane relative to a
    as b.
    :param a: projection base
    :param b: vector to project
    :return: a vector perpendicular to a, in plane with a and b, and with length |b|sin(theta)
    """
    # Calculate this by subtracting the projection from the original vector
    return b-vproj(a,b)


class Cart:
    def __init__(self, rail:Rail, u0:float, dsdt0:float, g:np.array, crr:float=0.0, kd:float=0.0):
        """
        Construct the mover
        :param rail: Rail that the mover will move along
        :param u0: Initial rail parameter [1]
        :param dsdt0: Initial cart scalar speed [m/s]. Note that this is in
                   physical units, not parameter/second. We refer to scalar
                   speed as ds/dt, and reserve v for vector velocity.
        :param g: Acceleration of gravity [m/s**2]. A free-falling object will accelerate
                   in the direction of g, so g defines "down". Magnitude determines strength
                   of gravity
        :param crr: Coefficient of rolling resistance, including inelastic deformation of the
           wheels, sticky grease in the bearings, rolling energy converted to sound, etc.
           The formula is Frr=crr*W, so units are [N/N]=[1]. Typical values for roller coaster is
           about 0.01 [https://www.coaster101.com/2011/10/24/coasters-101-wheel-design/]
           meaning 1000N of contact force will produce 10N of rolling resistance. Equivalently
           it will take 1lb of pull force to maintain speed of a 100lb cart on a level rail.
        :param kd: Drag constant [1/m]. This rolls up all the factors in brackets in the formula
          ad=-vhat/2*(rho*Cd*A/m)*v**2.
          [m/s**2]=[1]*[x]*[m**2/s**2]
          [m/s**2][s**2/m**2]=[x]
          [m][1/m**2]=[x]
          [1/m]=[x] s
          A single-car roller coaster might weigh 1000kg, have a Cd of about 1.8,
          a frontal area of about 1m**2, and operate in air of density about
          1.2kg/m**3. Multiplying these all together gives:
            1.2 kg/m**3*1.8*1 m**2/1000 kg=1.2*1.8*1/2000=0.00216/m with units
            [kg/m**3][m**2][1/kg]=[1/m**3][m**2]=[1/m] as expected.
        """
        self.rail=rail
        self.u=u0 # initial rail parameter corresponding to initial position on track
        self.dsdt=dsdt0 # scalar speed along track in [m/s]
        self.g=g # gravity vector in [m/s**2]
        self.crr=crr # mass of mover times coefficient of rolling resistance [kg]
        self.kd=kd
    def a(self, verbose=True)->np.array:
        """
        Tangential and normal acceleration of cart, in world frame
        :return: Tuple of:
         * Tangential acceleration. This is a scalar, and is the component
           of the acceleration along the track. If positive, the cart is
           accelerating in the +u direction, and if negative, it is
           slowing down (if it is moving forward) or accelerating backward
           (if already moving backward).
         * Acceleration vector. This could be integrated to get the velocity
           and position of the cart. Note that if there is gravity, an
           accelerometer wouldn't feel it, so an IMU simulator would need
           to subtract gravity from this vector.

        Only tangential acceleration affects the speed of the cart,
        but an accelerometer feels both. An accelerometer on a cart
        on a circular rail feels both centrepital acceleration and
        gravity, so it reports the vector sum of the inward centrepital
        acceleration and upward gravity acceleration. An accelerometer
        on a cart that is boosted or braking feels the boost or brake.
        An accelerometer on a cart falling down a vertical rail feels
        both the downward tangential acceleration of the cart and
        the upward gravity acceleration, so it's a net zero.

        """
        # unit vector in direction of track and in direction of track curvature.
        T=self.rail.T(self.u)
        N=self.rail.N(self.u)
        # vector velocity
        v=T*self.dsdt
        # Figure out forces on cart
        a=v*0.0
        # gravity
        a_grav=self.g
        a+=a_grav
        # Support -- the acceleration required keep the
        # cart from falling through the rail, equal and opposite
        # to the component of gravity perpendicular to the rail
        a_support=-vperp(T,self.g)
        a+=a_support
        # centripetal acceleration -- the acceleration required to
        # force the cart to follow the rail. This is:
        # a_centrip=v**2/r
        #          =v**2*kappa
        # and towards the center of the osculating circle,
        # therefore in the direction of N. Note that it only
        # changes the direction of motion, never the speed.
        kappa=self.rail.kappa(self.u)
        if kappa>0:
            # If rail curves, calculate centripetal acceleration.
            a_centrip= N * self.dsdt ** 2 * self.rail.kappa(self.u)
        else:
            # If the rail is straight, self.rail.N is a normalized
            # zero vector, so all components are 0/0=NaN. We would
            # like to use NaN*0=0, but NaN*0=NaN so we need a
            # special case.
            a_centrip=v*0.0
        a+=a_centrip

        # Calculate the total force needed to keep the cart on the rail
        a_rail=a_support+a_centrip

        # Rolling resistance -- This is modeled as a force
        # with direction opposite the velocity and magnitude
        # proportional to the force exerted on the track. Since
        # the cart rolls, it doesn't exert any force along the
        # track, so the force exerted on the track is purely
        # perpendicular to it.
        #
        # Frr =-(v/|v|)*crr*|W|
        #     =-(v/|v|)*crr*|W|
        #   W =m*aN
        # |W| =m*|aN|
        # Frr =-(v/|v|)*crr*m*|aN|
        # marr=-(v/|v|)*crr*m*|aN|
        #  arr=-(v/|v|)*crr*|aN|
        #
        # so the acceleration is independent of the mass. This doesn't
        # seem to take into account things like sticky grease. I would
        # say that crr is a weak function of mass
        if self.dsdt!=0:
            a_rr=-vnormalize(v)*self.crr*vlength(a_rail)
        else:
            a_rr=v*0.0
        a+=a_rr

        # Air resistance is calculated from the usual velocity-squared model.
        # We could have a speed-dependent Cd, but we don't.
        # Fd =-(v/|v|)*rho/2*Cd*v**2*A
        # adm=-vhat/2*(rho*Cd*A)*v**2
        # ad =-vhat/2*(rho*Cd*A/m)*v**2
        #  kd=rho*Cd*A/m # drag constant, rolls up all physical constants
        # ad =-vhat/2*kd*v**2
        # vhat=v/vlength(v)
        # ad =-v/vlength(v)/2*kd*v**2
        # ad =-v**3*kd/(2*vlength(v))
        if self.dsdt!=0:
            a_d=-v**3*self.kd/(2*vlength(v))
        else:
            a_d=v*0.0
        a+=a_d

        # track boost/brake

        # Now what component of acceleration is actually along the direction of travel?
        # The rest is exerted on the track, and the track reacts by pushing back (3rd law)
        # and canceling it out. We are calculating the acceleration scalar in the tangent
        # direction, so asT. This is what is needed for the equations of motion.
        aT=vproj(T,a)
        # We want the scalar projection https://en.wikipedia.org/wiki/Scalar_projection.
        # This is given by the dot product of the two vectors, divided by the length
        # of the reference vector. Since the reference vector T is already length 1,
        # vdot does just what we want. It's the length of the aT vector above, but
        # can be negative if the acceleration is opposite the sense of +u parameter
        # IE "backward" along the track
        asT=vdot(a,T)
        j=0
        def plot_vector(v,name):
            nonlocal j
            color=plt.rcParams['axes.prop_cycle'].by_key()['color'][j]
            j+=1
            plt.quiver(0, 0, v[0,0], v[2,0], angles='xy', scale_units='xy', scale=1, color=color, label=name)
            print(f"{name}={v}")
        if verbose:
            plt.figure(f"Plot at u={self.u}")
            plot_vector(a_grav,'a_grav')
            plot_vector(a_support,'a_support')
            plot_vector(a_centrip,'a_centrip')
            plot_vector(T,'T')
            plot_vector(N,'N')
            print(f"{aT=}")
            print(f"{asT=}")
            plt.axis('equal')
            plt.xlim(-10,10)
            plt.ylim(-10,10)
            plt.legend()
            plt.pause(1)
            plt.show()
        return asT,a
    def step(self,dt:float,verbose=True)->np.array:
        asT,a=self.a(verbose=verbose)
        dv=asT*dt
        ds= self.dsdt * dt
        self.dsdt+=dv
        # du=ds*(du/ds)
        #   =ds/(ds/du)
        du=ds/self.rail.dsdu(self.u)
        self.u+=du
        return a



