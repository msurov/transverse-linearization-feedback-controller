from casadi import MX, Function
from car_trailers import car_trailers_dynamics
from transverse_linearization import get_trans_lin
import numpy as np
from lqr import lqr_ltv
from serdict import load, save
import matplotlib.pyplot as plt


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


def eval_linsys():
    traj = load_trajectory('traj-1.npy')
    ntrailers = traj['ntrailers']
    rhs = car_trailers_dynamics(ntrailers)
    linsys = get_trans_lin(rhs, traj)
    save('linsys.npy', linsys)


def main():
    traj = load_trajectory('traj-1.npy')
    linsys = load('linsys.npy')

    t = linsys['t']
    J = linsys['J']
    A = linsys['A']
    B = linsys['B']

    n,nxi,nu = B.shape
    nx = nxi + 1
    Qx = np.eye(nx)
    R = np.zeros((n, nu, nu))
    Q = np.zeros((n, nx-1, nx-1))

    for i in range(len(t)):
        Q[i] = J[i].T @ Qx @ J[i]
        R[i] = np.eye(nu)

    S = Q[-1]
    K,P = lqr_ltv(t, A, B, Q, R, S)
    lqr = {
        't': t,
        'K': K,
        'P': P
    }
    save('lqr.npy', lqr)


if __name__ == '__main__':
    eval_linsys()
    main()
