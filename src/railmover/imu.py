"""
Describe purpose of this script here

Created: 10/13/24
"""
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt

from railmover.mover import Cart
from railmover.rail import CircularRail, BSpline, LinearSpline

sim_return_type = namedtuple('sim_return_type', 'ts us rs vs dsdts accs')
class IMU:
    """
    Class that creates IMU measurements
    """
    def __init__(self,cart:Cart):
        self.cart=cart
    def sim(self,*,t0:float=0.0,t1:float,fps:int=128)->np.array:
        """
        Simulate IMU data from a cart rolling on a rail
        :param t0:
        :param t1:
        :param fps:
        :return: Named tuple including all truth data including acceleration
         * ts - Measurement times
         * us - Parameters at measurements
         * rs - Position in 3D space at measurements
         * vs - Velocity in 3D space at measurements
         * dsdts - Speed along rail at measurements
         * accs - actual acceleration in inertial space at measurements. An accelerometer
                on a stable platform would experience accs-self.cart.g0
        """
        dt=1.0/fps
        ts=np.arange(t0,t1,dt)
        dsdts=ts*0.0
        accs=np.zeros((3,ts.shape[0]))
        vs=accs*0.0
        rs=accs*0.0
        us=ts*0.0
        for i_t,t in enumerate(ts):
            verbose=False
            rs[:,None,i_t]=self.cart.rail.r(self.cart.u)
            dsdts[i_t]=self.cart.dsdt
            vs[:,None,i_t]=self.cart.rail.T(self.cart.u)*self.cart.dsdt
            us[i_t]=self.cart.u
            accs[:,None,i_t]=self.cart.step(dt=dt,verbose=verbose)
        verbose=True
        if verbose:
            plt.figure("World acceleration")
            plt.plot(ts,accs[0,:],label='ax')
            plt.plot(ts,accs[1,:],label='ay')
            plt.plot(ts,accs[2,:],label='az')
            plt.xlabel('time/s')
            plt.ylabel('acc/(m/s**2)')
            plt.legend()
            plt.figure("Speed")
            plt.plot(ts,dsdts,label='dsdt')
            plt.xlabel('time/s')
            plt.ylabel('speed/(m/s)')
            plt.figure("True Course")
            plt.plot(rs[0,:],rs[2,:],'-')
            plt.plot(rs[0,::fps//8],rs[2,::fps//8],'x')
            plt.plot(rs[0,::fps],rs[2,::fps],'x')
            plt.xlabel='x/m'
            plt.ylabel='z/m'
            plt.axis('equal')
            plt.figure("Height")
            plt.plot(ts,rs[2,:],'-')
            plt.xlabel='t/s'
            plt.ylabel='z/m'
            plt.figure("Rail parameter $u$")
            plt.plot(ts,us,'-')
            plt.xlabel='t/s'
            plt.ylabel='u'
            plt.show()
        return sim_return_type(ts=ts,us=us,rs=rs,vs=vs,dsdts=dsdts,accs=accs)
    def integrate(self,sim:sim_return_type):
        recon_vs=np.zeros((3,sim.ts.shape[0]))
        recon_rs=recon_vs*0.0
        dt=sim.ts[1]-sim.ts[0]
        recon_r=sim.rs[:,np.newaxis,0]
        recon_v=sim.vs[:,np.newaxis,0]
        for i,(t0,r0,v0,a0,a1) in enumerate(zip(sim.ts,
                               np.hsplit(sim.rs,len(sim.ts)),
                               np.hsplit(sim.vs,len(sim.ts)),
                               np.hsplit(sim.accs,len(sim.ts)),
                               np.hsplit(np.roll(sim.accs,1,axis=1),len(sim.ts)))):
            recon_rs[:,np.newaxis,i]=recon_r
            recon_vs[:,np.newaxis,i]=recon_v
            j0=(a1-a0)/dt
            recon_a1=a0+j0*dt
            recon_v1=v0+a0*dt+j0*dt**2/2.0
            recon_r1=r0+v0*dt+a0*dt**2/2+j0*dt**3/6.0
            recon_r=recon_r1
            recon_v=recon_v1
        plt.figure('Trajectory')
        plt.plot(sim.rs[0,:],sim.rs[2,:],label='truth')
        plt.plot(recon_rs[0,:],recon_rs[2,:],'--',label='recon')
        plt.axis('equal')
        plt.legend()
        plt.figure('Height')
        plt.plot(sim.ts,sim.rs[2,:],label='truth')
        plt.plot(sim.ts,recon_rs[2,:],label='recon')
        plt.legend()
        plt.figure('dHeight')
        plt.plot(sim.ts,sim.rs[2,:]-recon_rs[2,:],label='diff')
        plt.legend()
        plt.figure('hdot')
        plt.plot(sim.ts,sim.vs[2,:],label='truth')
        plt.plot(sim.ts,recon_vs[2,:],label='recon')
        plt.legend()
        plt.figure('dhdot')
        plt.plot(sim.ts,sim.vs[2,:]-recon_vs[2,:],label='diff')
        plt.legend()
        plt.show()


def main():
    if True:
        rail=BSpline(np.array([
            [-200.0,-10.0, 10.0,  0.0,-10.0, 10.0,200.0],
            [   0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
            [ 200.0,-30.0,  0.0, 10.0,  0.0,-30.0,200.0],
                               ]))
        t1=10.27
        g=9.80665
    else:
        rail=LinearSpline(np.array([
            [0.0,0.0],
            [0.0,0.0],
            [1.0,0.0]]))
        t1=1.4142135
        g=1.0
    cart=Cart(rail=rail,
             u0=0,
             dsdt0=0.0,
             g=np.array([[0.0],[0.0],[-g]]),
             crr=0.0,
             kd=0.00
             )
    imu=IMU(cart)
    sim_result=imu.sim(t0=0.0,t1=t1,fps=128)
    imu.integrate(sim_result)



if __name__ == "__main__":
    main()
