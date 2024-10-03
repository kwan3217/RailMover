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

    s: actual distance along rail from r(0), m
    t: time from epoch, s. Not used in this rail, but reserved for rail mover
    u: rail parameter. Not the usual t, since that is reserved for actual time.
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
        :param u: Rail parameter
        :return: Position of rail as a vector with components in meters
        """
        raise NotImplementedError

    def drdu(self, u: float, du: float = 1e-4) -> np.array:
        """
        Differential change in position with respect to parameter
        :param u: Rail parameter
        :param du: Differential to use. Ignored if we are using
                   calculus to use a true infinitesimal differential.
        :return:
        """
        return self.r(u + du) - self.r(u)

    def d2rdu2(self, u: float, du: float = 1e-4) -> np.array:
        """
        Second Differential change in position with respect to parameter
        :param u:
        :param du:
        :return:
        """
        return self.drdu(u + du) - self.drdu(u)

    def d3rdu3(self, u: float, du: float = 1e-4) -> np.array:
        """
        Third ifferential change in position with respect to parameter
        :param u:
        :param du:
        :return:
        """
        return self.d2rdu2(u + du) - self.d2rdu2(u)

    def ds(self, u: float, du: float = 1e-4) -> float:
        """
        Second differential distance, numerator of second derivative (d**2)s/du**2
        If you have an explicit second derivative, multiply that result by
        du**2
        :param u: Rail parameter
        :param du: Differential parameter
        :return: Second differential distance, m**2
        """
        return vlength(self.drdu(u, du))

    def step_s(self, u: float, s: float):
        """
        Calculate the parameter which is a distance s away from a given
        parameter u
        :param u: Initial rail parameter
        :param s: Distance along rail. Positive S should give a u1 greater than u0,
          and conversely.
        :return: Parameter which is the given distance s along the rail.
        """
        raise NotImplementedError

    def T(self, u: float, du: float = 1e-4) -> np.array:
        """
        Calculate tangent vector to curve at given parameter. This vector
        will be unit-length and parallel to the derivative of the curve
        with respect to the parameter. If you have an explicit closed form,
        override this.
        :param u: Rail parameter
        :param du:
        :return: Normalized tangent vector
        """
        # Calculated with Frenet-Serret apparatus, as given
        # at https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas
        # and valid only for 2D or 3D curves
        return self.drdu(u, du) / self.ds(u, du)

    def dT(self, u: float, du: float = 1e-4) -> np.array:
        return self.T(u + du) - self.T(u)

    def N(self, u: float, du: float = 1e-4) -> np.array:
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
        raise NotImplementedError

    def B(self, u: float, du: float = 1e-4) -> np.array:
        """
        Calculate the binormal vector, perpendicular to the
        instantaneous plane of the curve
        :param u:
        :param du:
        :return:
        """
        return vcross(self.T(u,du),self.N(u,du))

    def kappa(self, u: float, du: float = 1e-4) -> float:
        """
        Calculate the curvature. Intuitively, curvature is the measure
        of the curve's departure from linearity. Naturally a line would
        have zero curvature. A circle is defined to have a curvature
        equal to the inverse of its radius. Any arbitrary differentiable
        curve has a curvature at each point, which matches the curvature of
        the osculating circle at that point. One way to interpret the units
        of curvature is rad/m, indicating that the curve deviates so many
        radians from its original path, for each meter along the path.

        If the curvature is positive, then the center of curvature
        is on the same side of the curve as N, and if negative, the opposite.
        :param u:
        :return: Curvature in rad/m. Inverse of kappa is radius of curvature, m

        Note -- default implementation is only well-defined for 2D and 3D curves.
        """
        drdu = self.drdu(u, du) / du
        d2rdu2 = self.d2rdu2(u, du) / du ** 2

        return vcross(drdu, d2rdu2) / vlength(drdu) ** 3
    def tau(self, u: float, du: float = 1e-4) -> float:
        """
        Calculate the torsion. Intuitively for a curve through
        3D space, torsion is the measure of the curve's departure
        from planarity. The concrete definition is the speed of
        rotation of the binormal vector B=T x N with respect to
        arc length. A 2D curve has a torsion of zero along its
        whole length.
        :param u:
        :param du:
        :return:
        """
        # Formula from https://en.wikipedia.org/wiki/Torsion_of_a_curve#Alternative_description
        drxddr=vcross(self.drdu(u, du), self.d2rdu2(u, du))
        return vdot(drxddr, self.d3rdu3(u, du))/vlength(drxddr)**2


class CubicRail(Rail):
    """
    This is a rail which uses cubic parametric polynomials expressed in
    terms of a characteristic matrix.

    r=R @ B @ T
    R - matrix of control points as column vectors
    B - characteristic matrix
    T - parameter column vector
    """
    def r(self, u: float|np.array) -> np.array:
        """
        Position vector as a function of rail parameter.
        :param u: Rail parameter
        :return: Position of rail as a vector with components in meters
        """
        is=self.select_i(u)
        T=self.calcT(u)
        raise NotImplementedError


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

