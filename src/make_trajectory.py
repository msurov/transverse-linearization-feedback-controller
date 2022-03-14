from re import L
from casadi import MX, DM, Function, jacobian, sqrt, arctan2, sin, cos, pi
from car_trailers import car_trailers_dynamics
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from serdict import load, save


def eval_vars(x : Function, y : Function):
    t = MX.sym('t')
    xexpr = x(t)
    yexpr = y(t)
    dx = jacobian(xexpr, t)
    dy = jacobian(yexpr, t)
    u1 = sqrt(dx**2 + dy**2)
    theta0 = arctan2(dy, dx)
    dtheta0 = jacobian(theta0, t)
    phi = arctan2(dtheta0, u1)
    dphi = jacobian(phi, t)
    u2 = dphi
    theta0_fun = Function('theta0', [t], [theta0])
    phi_fun = Function('phi', [t], [phi])
    u1_fun = Function('u1', [t], [u1])
    u2_fun = Function('u2', [t], [u2])
    return {
        'u1': u1_fun,
        'u2': u2_fun,
        'phi': phi_fun,
        'theta0': theta0_fun
    }

def simulate(u1fun, u2fun, tspan, st1):
    ntrailers = len(st1) - 3
    ct = car_trailers_dynamics(ntrailers)

    def rhs(t, x):
        u = np.reshape([u1fun(t), u2fun(t)], (-1,))
        dx = ct(x, u)
        return np.reshape(dx, (-1,))

    ans = solve_ivp(rhs, [tspan[0], tspan[-1]], st1, max_step=1e-3)
    assert ans.success
    return ans.t, ans.y.T


def sample_traj_1():
    t = MX.sym('t')
    xfun = Function('x', [t], [sin(4 * pi * t)])
    yfun = Function('y', [t], [cos(6 * pi * t)])
    vars = eval_vars(xfun, yfun)
    
    st0 = [0, 0, float(vars['phi'](0)), float(vars['theta0'](0)), 0, 0]
    tspan = [0, 1]
    t, st = simulate(vars['u1'], vars['u2'], tspan, st0)
    st0 = st[-1]
    t, st = simulate(vars['u1'], vars['u2'], tspan, st0)

    u1 = np.reshape(DM(vars['u1'](t)), (-1,))
    u2 = np.reshape(DM(vars['u2'](t)), (-1,))

    traj = {
        'ntrailers': st.shape[1] - 3,
        't': t,
        'x': st[:,0],
        'y': st[:,1],
        'phi': st[:,2],
        'theta': st[:,3:].T,
        'u1': u1,
        'u2': u2
    }
    return traj


def reverse_trajectory(traj):
    t = traj['t']
    return {
        'ntrailers': traj['ntrailers'],
        't': t[-1] - t[::-1],
        'x': traj['x'][::-1],
        'y': traj['y'][::-1],
        'phi': traj['phi'][::-1],
        'theta': traj['theta'][:,::-1],
        'u1': -traj['u1'][::-1],
        'u2': -traj['u2'][::-1],
    }

def sample_traj_2():
    traj = sample_traj_1()
    traj = reverse_trajectory(traj)
    save('traj.npy', traj)


sample_traj_2()
