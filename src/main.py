from casadi import MX, Function
from car_trailers import car_trailers_dynamics
from transverse_linearization import get_trans_lin
import numpy as np
from lqr import lqr_ltv, lqr_ltv_periodic
from serdict import load, save
import matplotlib.pyplot as plt
from transverse_feedback import TransverseFeedback
from scipy.integrate import solve_ivp
from make_trajectory import sample_traj_2
from simulator import Simulator


def load_trajectory(filename : str):
    data = load(filename)
    t = data['t']
    ntrailers = data['ntrailers']
    state = np.zeros((len(t), ntrailers + 3))
    state[:,0] = data['x']
    state[:,1] = data['y']
    state[:,2] = data['phi']
    state[:,3:] = data['theta'].T
    ctrl = np.zeros((len(t), 2))
    ctrl[:,0] = data['u1']
    ctrl[:,1] = data['u2']
    return {
        'ntrailers': ntrailers,
        't': t,
        'x': state,
        'u': ctrl
    }


def make_linsys(trajfile : str, linsysfile : str):
    traj = load_trajectory(trajfile)
    ntrailers = traj['ntrailers']
    rhs = car_trailers_dynamics(ntrailers)
    linsys = get_trans_lin(rhs, traj)
    save(linsysfile, linsys)


def make_lqr(linsysfile : str, lqrfile : str):
    linsys = load(linsysfile)

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
    save(lqrfile, lqr)


def run_simulation(trajfile : str, linsysfile : str, lqrfile : str, simfile : str):
    np.random.seed(0)

    traj = load_trajectory(trajfile)
    linsys = load(linsysfile)
    lqr = load(lqrfile)
    fb = TransverseFeedback(traj, lqr, linsys)
    ct_rhs = car_trailers_dynamics(traj['ntrailers'])

    sysrhs = lambda _x, _u: np.reshape(ct_rhs(_x, _u), (-1,))
    sim = Simulator(sysrhs, fb, 5e-4)
    st0 = traj['x'][0].copy()
    st0 += 0.005 * np.random.normal(size=st0.shape)
    t1 = traj['t'][0]
    t2 = traj['t'][-1]
    result = sim.run(st0, t1, t2)

    t, st, u, fb = result
    tau,xi = zip(*fb)

    simdata = {
        'ntrailers': traj['ntrailers'],
        't': t,
        'x': st[:,0],
        'y': st[:,1],
        'phi': st[:,2],
        'theta': st[:,3:].T,
        'u1': u[:,0],
        'u2': u[:,1],
        'tau': tau,
        'xi': xi
    }
    save(simfile, simdata)


if __name__ == '__main__':
    trajfile = 'traj.npy'
    linsysfile = 'linsys.npy'
    lqrfile = 'lqr.npy'
    simfile = 'sim.npy'
    # sample_traj_2(trajfile)
    # make_linsys(trajfile, linsysfile)
    make_lqr(linsysfile, lqrfile)
    run_simulation(trajfile, linsysfile, lqrfile, simfile)
