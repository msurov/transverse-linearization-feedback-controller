from casadi import MX, Function
from car_trailers.dynamics import car_trailers_dynamics
from transverse_linearization.transverse_linearization import get_trans_lin
import numpy as np
from common.lqr import (
    lqr_ltv, lqr_ltv_periodic
)
import matplotlib.pyplot as plt
from transverse_linearization.transverse_feedback import TransverseFeedback
from scipy.integrate import solve_ivp
from car_trailers_traj_planner.planner import (
    sample_traj_2,
    sample_traj_1
)
from common.simulator import Simulator
from car_trailers.anim import animate
from dataclasses import dataclass
from car_trailers_traj_planner.trajectory import CarTrailersTrajectory


def make_linsys(trajfile : str, linsysfile : str):
    traj = np.load(trajfile, allow_pickle=True).item()
    ntrailers = traj.ntrailers
    rhs = car_trailers_dynamics(ntrailers)
    linsys = get_trans_lin(rhs, traj)
    np.save(linsysfile, linsys)


def make_lqr(linsysfile : str, lqrfile : str):
    linsys = np.load(linsysfile, allow_pickle=True).item()

    t = linsys['t']
    J = linsys['J']
    A = linsys['A']
    B = linsys['B']

    n,nxi,nu = B.shape
    nx = nxi + 1

    Qx = 1000 * np.eye(nx)
    Qx[0,0] = \
    Qx[1,1] = 20
    Qx[2,2] = 20

    R = np.zeros((n, nu, nu))
    Q = np.zeros((n, nx-1, nx-1))

    for i in range(len(t)):
        Q[i] = J[i].T @ Qx @ J[i]
        R[i] = np.eye(nu)

    periodic = np.allclose(linsys['A'][0], linsys['A'][-1]) and \
        np.allclose(linsys['B'][0], linsys['B'][-1])

    if periodic:
        K,P = lqr_ltv_periodic(t, A, B, Q, R)
    else:
        S = Q[-1]
        K,P = lqr_ltv(t, A, B, Q, R, S)

    lqr = {
        't': t,
        'K': K,
        'P': P
    }
    np.save(lqrfile, lqr)

@dataclass
class SimulatedTrajectory(CarTrailersTrajectory):
    tau : np.ndarray
    xi : np.ndarray

    def __post_init__(self):
        super().__post_init__()
        self.tau = np.array(self.tau, float)
        self.xi = np.array(self.xi, float)
        n, = self.tau.shape
        assert self.xi.shape[0] == n

def run_simulation(
        trajfile : str,
        linsysfile : str,
        lqrfile : str,
        simfile : str) -> SimulatedTrajectory:

    np.random.seed(0)

    traj = np.load(trajfile, allow_pickle=True).item()
    linsys = np.load(linsysfile, allow_pickle=True).item()
    lqr = np.load(lqrfile, allow_pickle=True).item()
    fb = TransverseFeedback(traj, lqr, linsys)
    ct_rhs = car_trailers_dynamics(traj.ntrailers)

    sysrhs = lambda _x, _u: np.reshape(ct_rhs(_x, _u), (-1,))
    sim = Simulator(sysrhs, fb, 5e-4)
    st0 = traj.state[0].copy()
    st0 += 0.005 * np.random.normal(size=st0.shape)
    t1 = traj.time[0]
    t2 = traj.time[-1]
    result = sim.run(st0, t1, t2)

    t, st, u, fb = result
    tau,xi = zip(*fb)
    traj = SimulatedTrajectory(
        time = t,
        state = st,
        control = u,
        tau = tau,
        xi = xi
    )
    np.save(simfile, traj)

def run_animation(trajfile, simfile, writetofile=None):
    ref_traj = np.load(trajfile, allow_pickle=True).item()
    real_traj = np.load(simfile, allow_pickle=True).item()
    if writetofile is not None:
        animate(ref_traj, real_traj, animtime=20, filepath=writetofile)
    else:
        animate(ref_traj, real_traj, animtime=20)

def main():
    trajfile = 'data/traj.npy'
    linsysfile = 'data/linsys.npy'
    lqrfile = 'data/lqr.npy'
    simfile = 'data/sim.npy'
    traj = sample_traj_2()
    np.save(trajfile, traj)
    make_linsys(trajfile, linsysfile)
    make_lqr(linsysfile, lqrfile)
    run_simulation(trajfile, linsysfile, lqrfile, simfile)
    run_animation(trajfile, simfile, 'data/anim.gif')

if __name__ == '__main__':
    main()
