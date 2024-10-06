"""
Simulate movement along a rail, such as on a roller-coaster
"""

import numpy as np
from kwanmath.vector import vlength, vnormalize, vcross, vdot


class Rail:
    """
    Rail is a parametric function of an arbitrary parameter, related to
    but not necessarily proportional to distance along rail. Input is
    scalar-valued parameter, output is position vector. This is to
    support such things as Bezier curves where it is difficult to
    have an explicit form for distance.

    We use the following common terms. Units are documented as the
    appropriate SI units, but as is always the case, you are free to
    use whatever set of consistent physical units you want.

    s: actual distance along rail from r(0), [m]
    t: time from epoch, [s]. Not used in this rail, but reserved for rail mover
    u: rail parameter, [1]. Not the usual t, since that is reserved for actual time.
    u0, u1: Domain of rail parameter, for instance 0 to 1 in a Bezier curve segment.
       Values outside this range request extrapolation, and the implementation must
       either do the extrapolation or raise ValueError. It is strongly suggested
       that if extrapolation is possible, to implement it on all functions of
       parameter. However sometimes this is not possible (for instance you can
       calculate a position but not a tangent extrapolation) and in this case
       some functions will work and others will raise exceptions.
    s0, s1: range of rail distances. The definition of s implies that s0==0
       for all rails, and s1 is the total length of the rail along its domain.
    dt: differential time, s
    du: differential parameter
    """

    def r(self, u: float) -> np.array:
        """
        Position vector as a function of rail parameter.
        :param u: Rail parameter [1]
        :return: Position of rail as a vector [m]

        All curves need to override this method
        """
        raise NotImplementedError

    def dnrdun(self, u: float, n:int) -> np.array:
        """
        Arbitrary derivative of position with respect to parameter

        :param u: Rail parameter [1]
        :param n: Order of derivative. N=0 returns the function itself,
                  so the zeroth derivative is the function itself.
                  Negative orders are not allowed.
        :param du: Differential to use. Ignored if we are using
                   calculus to use a true infinitesimal differential.
        :return: vector derivative, [m/1]=[m]

        Curves with an explicit form for the derivative should
        override this method
        """
        raise NotImplementedError

    def dsdu(self, u: float) -> float:
        """
        Derivative of distance along rail with respect to parameter

        :param u: Rail parameter
        :return: ds/du, derivative of distance along rail with respect to parameter

        This function should NOT be overridden -- it is explicit and exact if
        self.dnrdun(u,1) is explicit.
        """
        # This derivative is just the length of dr/du
        return vlength(self.dnrdun(u, 1))

    def step_s(self, u: float, s: float, quality:float):
        """
        Calculate the parameter which is a distance s away from a given
        parameter u
        :param u: Initial rail parameter
        :param s: Distance along rail. Positive S should give a u1 greater than u0,
          and conversely.
        :return: Parameter which is the given distance s along the rail.
        """
        raise NotImplementedError

    def T(self, u: float) -> np.array:
        """
        Calculate tangent vector to curve at given parameter. This vector
        will be unit-length and parallel to the derivative of the curve
        with respect to the parameter. If you have an explicit closed form,
        override this.
        :param u: Rail parameter
        :param du:
        :return: Normalized tangent vector, unitless

        Should NOT be overridden. This is exact if dnrdun(u,1) is explicit
        """
        # Calculated with Frenet-Serret apparatus, as given
        # at https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas
        # and valid only for 2D or 3D curves
        # The given formula is T=dr/ds. We have dr/du and ds/du, so we have:
        # T=dr/ds
        #  =(dr/du)*(du/ds) # chain rule
        #  =(dr/du)/(ds/du) # treat derivative as a fraction
        n=self.dnrdun(u, 1)
        d=self.dsdu(u)
        return  n/d

    def dTdu(self,u:float)->np.array:
        """
        Calculate the derivative of the tangent vector.
        """
        # We will need the first and second derivatives of
        # the curve with respect to the parameter
        rp=self.dnrdun(u,1)
        rpp=self.dnrdun(u,2)
        # We start with
        # T(u)=r'(u)/s'(u)
        # and want to find the derivative. We need the quotient rule, where:
        #   h(u)=f(u)/g(u)
        #         g(u)f'(u)-f(u)g'(u)
        #   h'(u)=-------------------
        #             (g(u))**2
        # Here, we have:
        #   h(u)=T(u)=r'(u)/s'(u)
        # so we have:
        #   f(u)=r'(u)    which is given
        f=rp
        #   f'(u)=r''(u)  which is given
        fp=rpp
        #   g(u)=s'(u)    which is given
        g=self.dsdu(u)
        #       =sqrt(r'(u).r'(u))  expanded definition of s'(u)=|r'(u)|
        #   g'(u)=s''(u)
        # which we need to figure out with the chain rule. If we define:
        #   p(v)=v.v
        #       =r'.r'
        p=vdot(rp,rp)
        #   q(p)=sqrt(p)
        # we then have:
        #   dq/dv=dq/dp*dp/dv
        # or in prime notation:
        #   q'(v)=q'(p(v))*p'(v)
        # p'(v) is from the product rule and is:
        #   p'=2*v.v'
        # so in our case since v=r', we have:
        #   p'=2*r'.r''
        pp=2*vdot(rp,rpp)
        # Now q'(p)=1/(2*sqrt(p)) by the power rule
        qp=1/(2*np.sqrt(p))
        # so g'(u)=q'(u)=q'(p)*p'(u) by the chain rule
        gp=qp*pp
        # Now we have all the parts and can use the quotient rule
        hpn=g*fp-f*gp
        hpd=g**2
        hp=hpn/hpd
        return hp

    def N(self, u: float) -> np.array:
        """
        Calculate the normal vector. Should always be directed such that N(u)/kappa(u)
        points from the curve to the instantaneous center of curvature. In 2D, it's probably
        a good idea to have N always be a fixed 90deg rotation from T, so that the sign
        of kappa determines whether you are turning left or right. In higher dimensions,
        this may be more difficult to keep consistent, so just keep the constraint
        mentioned above.
        :param u:
        :return:
        """
        # dT/ds=(dT/du)*(du/ds) # chain rule
        #      =(dT/du)/(ds/du) # treat derivative as a fraction
        dTds=self.dTdu(u)/self.dsdu(u)
        return vnormalize(dTds)

    def B(self, u: float) -> np.array:
        """
        Calculate the binormal vector, perpendicular to the
        instantaneous plane of the curve
        :param u:
        :param du:
        :return:
        """
        return vcross(self.T(u),self.N(u))

    def kappa(self, u: float) -> float:
        """
        Calculate the curvature. Intuitively, curvature is the measure
        of the curve's departure from linearity. Naturally a line would
        have zero curvature. A circle is defined to have a curvature
        equal to the inverse of its radius. Any arbitrary differentiable
        curve has a curvature at each point, which matches the curvature of
        the osculating circle at that point. One way to interpret the units
        of curvature is rad/m, indicating that the curve deviates so many
        radians from its original path, for each meter along the path.

        Curvature is always positive, and the center of curvature
        is on the same side of the curve as N points.
        :param u:
        :return: Curvature in rad/m. Inverse of kappa is radius of curvature, m

        """
        if False:
            # Frenet-Serret formula
            dTds=self.dTdu(u,du)/self.dsdu(u,du)
            return vlength(dTds)
        else:
            # Direct using dot-product formula at
            # https://en.wikipedia.org/wiki/Curvature#General_expressions
            # valid for any dimension of vector
            du1=self.dnrdun(u,1)
            du2=self.dnrdun(u,2)
            sq1a=vlength(du1)**2
            sq1b=vlength(du2)**2
            sq1=sq1a*sq1b
            sq2=vdot(du1,du2)**2
            n=np.sqrt(sq1-sq2)
            d=vlength(du1)**3
            return n/d


    def tau(self, u: float) -> float:
        """
        Calculate the torsion. Intuitively for a curve through
        3D space, torsion is the measure of the curve's departure
        from planarity. The concrete definition is the speed of
        rotation of the binormal vector K=T x N with respect to
        arc length. A 2D curve has a torsion of zero along its
        whole length.
        :param u:
        :param du:
        :return:
        """
        # Formula from https://en.wikipedia.org/wiki/Torsion_of_a_curve#Alternative_description
        du1=self.dnrdun(u, 1)
        du2=self.dnrdun(u, 2)
        du3=self.dnrdun(u, 3)
        n1=vcross(du1, du2)
        n=vdot(n1,du3)
        d=vdot(n1,n1)
        return n/d

    def calc_u(self, s:float):
        if s==0.0:
            return 0.0
        raise NotImplementedError("Todo - curve arc length stuff")

    def plot_frame(self,u0,u1,du,step):
        import matplotlib.pyplot as plt
        fig=plt.figure()
        xyz=fig.add_subplot(224,projection='3d')
        xy=fig.add_subplot(221)
        yz=fig.add_subplot(223)
        xz=fig.add_subplot(222)
        us = np.arange(u0, u1, du)
        r = self.r(us)
        x=r[0,:]
        y=r[1,:]
        z=r[2,:]
        xyz.plot(x,y,z)
        xyz.set_xlabel('x')
        xyz.set_ylabel('y')
        xyz.set_zlabel('z')
        xy.plot(x,y)
        xy.set_xlabel('x')
        xy.set_ylabel('y')
        xy.axis('equal')
        yz.plot(y,z)
        yz.set_xlabel('y')
        yz.set_ylabel('z')
        yz.axis('equal')
        xz.plot(x,z)
        xz.set_xlabel('x')
        xz.set_ylabel('z')
        xz.axis('equal')
        for u in us[::step]:
            r0 = self.r(u)
            N = self.N(u)
            r1 = r0 + N
            r = np.hstack((r0, r1))
            xyz.plot(r[0, :], r[1, :], r[2, :], 'g-')
            xy.plot(r[0, :], r[1, :], 'g-')
            yz.plot(r[1, :], r[2, :], 'g-')
            xz.plot(r[0, :], r[2, :], 'g-')

            T = self.T(u)
            r1 = r0 + T
            r = np.hstack((r0, r1))
            xyz.plot(r[0, :], r[1, :], r[2, :], '-', color='#c0c000')
            xy.plot(r[0, :], r[1, :], '-', color='#c0c000')
            yz.plot(r[1, :], r[2, :], '-', color='#c0c000')
            xz.plot(r[0, :], r[2, :], '-', color='#c0c000')

            B = self.B(u)
            r1 = r0 + B
            r = np.hstack((r0, r1))
            xyz.plot(r[0, :], r[1, :], r[2, :], 'b-')
            xy.plot(r[0, :], r[1, :], 'b-')
            yz.plot(r[1, :], r[2, :], 'b-')
            xz.plot(r[0, :], r[2, :], 'b-')
        xyz.set_aspect('equal')
        plt.show()


class NDerivRail(Rail):
    """
    Rail where the derivative needs to be calculated numerically. Don't intend
    to ever actuall need this, we just keep the code for numerical derivative.
    """
    def __init__(self,du:float=1e-4):
        self.du=du
    def dnrdun(self, u: float, n:int) -> np.array:
        """
        Arbitrary derivative of position with respect to parameter

        :param u: Rail parameter [1]
        :param n: Order of derivative. N=0 returns the function itself,
                  so the zeroth derivative is the function itself.
                  Negative orders are not allowed.
        :param du: Differential to use. Ignored if we are using
                   calculus to use a true infinitesimal differential.
        :return: vector derivative, [m/1]=[m]

        Curves with an explicit form for the derivative should
        override this method
        """
        if n==0:
            return self.r(u)
        else:
            return (self.dnrdun(u+self.du,n-1)-self.dnrdun(u,n-1))/self.du


class CircularRail(Rail):
    def __init__(self, radius):
        self.radius = radius

    def r(self, u: float) -> np.array:
        """
        Position vector of a point on the circle.

        :param u: Angle in radians [1]
        :return: Position vector [m]
        """
        return np.array([[self.radius * np.cos(u)], [self.radius * np.sin(u)], [0.0*u]])

    def dnrdun(self, u: float, n: int, du: float = 1e-4) -> np.array:
        """
        Explicit nth derivative of the position vector for a circular rail.

        :param u: Angle in radians [1]
        :param n: Order of the derivative
        :param du: Differential angle (ignored for explicit calculation)
        :return: nth derivative of the position vector [m/rad^n]
        """
        if n % 4 == 0:
            return self.r(u)
        elif n % 4 == 1:
            return np.array([[-self.radius * np.sin(u)], [self.radius * np.cos(u)],[0.0]])
        elif n % 4 == 2:
            return np.array([[-self.radius * np.cos(u)], [-self.radius * np.sin(u)],[0.0]])
        else:
            return np.array([[self.radius * np.sin(u)], [-self.radius * np.cos(u)],[0.0]])


class HelicalRail(Rail):
    def __init__(self, pitch=1, radius=1):
        self.pitch = pitch  # Distance the helix rises per revolution
        self.radius=radius

    def r(self, u: float) -> np.array:
        """
        Position vector of a point on the helix.

        :param u: Angle in radians [1]
        :return: Position vector [m]
        """
        return np.array([[self.radius*np.cos(u)], [self.radius*np.sin(u)], [self.pitch * u / (2 * np.pi)]])

    def dnrdun(self, u: float, n: int) -> np.array:
        """
        Explicit nth derivative of the position vector for a helical rail.

        :param u: Angle in radians [1]
        :param n: Order of the derivative
        :param du: Differential angle (ignored for explicit calculation)
        :return: nth derivative of the position vector [m/rad^n]
        """
        if n == 0:
            return self.r(u)
        elif n == 1:
            return np.array([[-self.radius*np.sin(u)], [ self.radius*np.cos(u)], [0*u+self.pitch / (2 * np.pi)]])
        elif n == 2:
            return np.array([[-self.radius*np.cos(u)], [-self.radius*np.sin(u)], [0*u]])
        elif n == 3:
            return np.array([[ self.radius*np.sin(u)], [-self.radius*np.cos(u)], [0*u]])
        else:
            raise NotImplemented("Fourth or higher derivative")



class CubicRail(Rail):
    """
    This is a rail which uses cubic parametric polynomials expressed in
    terms of a characteristic matrix.

    r=R @ K @ U
    R - matrix of control points as column vectors
    K - characteristic matrix
    U - parameter column vector
    """
    def select_i(self,u:float):
        try:
            return set([int(i) for i in np.clip(np.floor(u),self.min_i,self.max_i)])
        except TypeError:
            return {int(np.clip(np.floor(u),self.min_i,self.max_i))}
    def select_R(self,i:int):
        """This depends on the curve implementation"""
        raise NotImplementedError
    fn = [[lambda u: u * 0 + 1,  # direct
           lambda u: u,
           lambda u: u ** 2,
           lambda u: u ** 3],
          [lambda u: u * 0,  # first derivative
           lambda u: u * 0 + 1,
           lambda u: 2 * u,
           lambda u: 3 * u ** 2],
          [lambda u: u * 0,  # second derivative
           lambda u: u * 0,
           lambda u: u * 0 + 2,
           lambda u: 6 * u],
          [lambda u: u * 0,  # third derivative
           lambda u: u * 0,
           lambda u: u * 0,
           lambda u: u*0+6],
          [lambda u: u * 0,  # fourth and higher derivatives are all zero
           lambda u: u * 0,
           lambda u: u * 0,
           lambda u: u * 0],
          ]
    def calc_dnUdun(self,i:int,u:float,n:int):
        """
        Calculate the U matrix
        """
        # Handle higher than fourth derivative
        if n>=len(self.fn):
            n=len(self.fn)-1
        seg_u=u-i
        this_i = np.clip(np.floor(u),self.min_i,self.max_i).astype(int)
        if not isinstance(u,np.ndarray):
            if this_i==i:
                return np.array([[f(seg_u)] for f in self.fn[n]])
            else:
                return np.array([[],[],[],[]])
        w=(this_i==i)
        return np.vstack([f(seg_u[w]) for f in self.fn[n]])
    def calc_U(self,i:int,u:float):
        return self.calc_dnUdun(i,u,0)
    def dnrdun(self, u: float,n:int,du:float=1e-4) -> np.array:
        """
        Explicit derivative of position with respect to parameter

        :param u: Rail parameter [1]
        :param du: Differential to use. Ignored since we are using
                   calculus to use a true infinitesimal differential.
        :return: vector derivative, [m/1]=[m]

        In the case of a cubic, we have the defining equation

           r(u)=R @ K @ U(u)

        The derivative of any matrix with respect to a scalar is just
        the derivative of each cell with respect to that scalar. We
        therefore have:

           dr(u)/du = R @ K @ dU(u)/du

        Each component of U is just a polynomial function of u so the
        derivative is straightforward to calculate:

        U(u)=[1 u u**2 u**3]^T
        dU(u)/du=[0 1 2u 3u**2]
        """
        ii=self.select_i(u)
        result=[]
        for i in ii:
            R=self.select_R(i)
            U=self.calc_dnUdun(i,u,n)
            result.append(R @ self.K @ U)
        return np.hstack(result)
    def r(self, u: float) -> np.array:
        """
        Position vector as a function of rail parameter.
        :param u: Rail parameter
        :return: Position of rail as a vector with components in meters
        """
        return self.dnrdun(u,0)


class BSpline(CubicRail):
    K=np.array([[ 1.0,-3.0, 3.0,-1.0],
                [ 4.0, 0.0,-6.0, 3.0],
                [ 1.0, 3.0, 3.0,-3.0],
                [ 0.0, 0.0, 0.0, 1.0]])/6.0
    min_i=0
    min_u=0.0
    def __init__(self,control_points:np.array):
        """
        Create a BSpline
        """
        self.control_points=control_points
        self.max_i=control_points.shape[1]-4
        self.max_u=float(self.max_i)+1.0
    def select_R(self,i:int):
        return self.control_points[:,i:i+4]


class BezierRail(Rail):
    """
    Describe a Bezier rail
    """
    B   =[[lambda u: 1.0                                ],
          [lambda u: 1.0 - 1.0*u                        ,
           lambda u: 0.0 + 1.0*u                        ],
          [lambda u: 1.0 - 2.0*u +    1.0*u**2          ,
           lambda u: 0.0 + 2.0*u -    2.0*u**2          ,
           lambda u: 0.0 + 0.0*u +    1.0*u**2          ],
          [lambda u: 1.0 - 3.0*u +    3.0*u**2 -1.0*u**3,
           lambda u: 0.0 + 3.0*u -    6.0*u**2 +3.0*u**3,
           lambda u: 0.0 + 0.0*u +    3.0*u**2 -3.0*u**3,
           lambda u: 0.0 + 0.0*u +    0.0*u**2 +1.0*u**3]]
    dB  =[[lambda u: 0.0                                 ],
          [lambda u:     - 1.0                           ,
           lambda u:     + 1.0                           ],
          [lambda u:     - 2.0   +2.0*1.0*u              ,
           lambda u:     + 2.0   -2.0*2.0*u              ,
           lambda u:     + 0.0   +2.0*1.0*u              ],
          [lambda u:     - 3.0   +2.0*3.0*u -3.0*1.0*u**2,
           lambda u:     + 3.0   -2.0*6.0*u +3.0*3.0*u**2,
           lambda u:     + 0.0   +2.0*3.0*u -3.0*3.0*u**2,
           lambda u:     + 0.0   +2.0*0.0*u +3.0*1.0*u**2]]
    ddB =[[lambda u: 0.0                       ],
          [lambda u:     0.0                   ,
           lambda u:     0.0                   ],
          [lambda u:        +2.0*1.0           ,
           lambda u:        -2.0*2.0           ,
           lambda u:        +2.0*1.0           ],
          [lambda u:        +2.0*3.0 -6.0*1.0*u,
           lambda u:        -2.0*6.0 +6.0*3.0*u,
           lambda u:        +2.0*3.0 -6.0*3.0*u,
           lambda u:        +2.0*0.0 +6.0*1.0*u]]
    dddB=[[lambda u: 0.0                       ],
          [lambda u:     0.0                   ,
           lambda u:     0.0                   ],
          [lambda u:        +0.0           ,
           lambda u:        -0.0           ,
           lambda u:        +0.0           ],
          [lambda u:         -6.0*1.0,
           lambda u:         +6.0*3.0,
           lambda u:         -6.0*3.0,
           lambda u:         +6.0*1.0]]
    def __init__(self,r0,r1,r2,r3):
        self.r=[r0,r1,r2,r3]
    def r(self, u: float) -> np.array:
        result=self.r[0]*self.B[3][0]
        for i,this_r in self.r[1:]:
            result+=this_r*self.B[3][i+1]
        return result
    def drdu(self, u: float, du:float=1e-4)->np.array:
        result=self.r[0]*self.dB[3][0]
        for i,this_r in self.r[1:]:
            result+=this_r*self.dB[3][i+1]
        return result*du
    def d2rdu2(self, u: float, du:float=1e-4)->np.array:
        result=self.r[0]*self.dB[3][0]
        for i,this_r in self.r[1:]:
            result+=this_r*self.dB[3][i+1]
        return result*du**2
    def d3rdu3(self, u: float, du:float=1e-4)->np.array:
        result=self.r[0]*self.dB[3][0]
        for i,this_r in self.r[1:]:
            result+=this_r*self.dB[3][i+1]
        return result*du**3

