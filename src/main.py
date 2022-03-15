from casadi import MX, Function
from car_trailers import car_trailers_dynamics
from transverse_linearization import get_trans_lin
import numpy as np
from lqr import lqr_ltv, lqr_ltv_periodic
from serdict import load, save
import matplotlib.pyplot as plt
from transverse_feedback import TransverseFeedback
from scipy.integrate import solve_ivp


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


def make_linsys(trajfile):
    traj = load_trajectory(trajfile)
    ntrailers = traj['ntrailers']
    rhs = car_trailers_dynamics(ntrailers)
    linsys = get_trans_lin(rhs, traj)
    save('linsys.npy', linsys)


def make_lqr(trajfile):
    traj = load_trajectory(trajfile)
    linsys = load('linsys.npy')

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

    periodic = np.allclose(traj['x'][0], traj['x'][-1])
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
    save('lqr.npy', lqr)


def run_simulation(trajfile):
    traj = load_trajectory(trajfile)
    linsys = load('linsys.npy')
    lqr = load('lqr.npy')
    fb = TransverseFeedback(traj, lqr, linsys)
    ct_rhs = car_trailers_dynamics(traj['ntrailers'])

    def rhs(_, x):
        u = fb.process(x)
        dx = ct_rhs(x, u)
        return np.reshape(dx, (-1,))

    state0 = traj['x'][0].copy()
    np.random.seed(0)
    delta = 0.005 * np.random.normal(size=state0.shape)
    print(delta)
    state0 += delta
    t0 = traj['t'][0]
    t1 = traj['t'][-1]
    t_eval = np.linspace(t0, t1, 200)
    ans = solve_ivp(rhs, [t_eval[0], t_eval[-1]], state0, t_eval=t_eval, max_step=1e-3)
    assert ans.success
    state = ans.y.T

    plt.plot(traj['x'][:,0], traj['x'][:,1], '--')
    plt.plot(state[:,0], state[:,1])
    plt.show()

    traj = {
        'ntrailers': traj['ntrailers'],
        't': ans.t,
        'x': state[:,0],
        'y': state[:,1],
        'phi': state[:,2],
        'theta': state[:,3:].T
    }
    save('traj-sim.npy', traj)


if __name__ == '__main__':
    trajfile = 'traj.npy'
    # make_linsys(trajfile)
    make_lqr(trajfile)
    run_simulation(trajfile)
