from casadi import MX, Function, substitute, jacobian, Function, jtimes
import numpy as np
from scipy.interpolate import make_interp_spline
from construct_basis import construct_basis
from bsplinesx import bsplinesf


def get_trans_lin(dynamics_rhs : Function, trajectory : dict):
    R'''
        `dynamics_rhs` is a casadi MX Function representing 
            the right hand side of the dynamic system \
        `trajectory` is the dict with the following entries
            `t` is time knots of the trajectory
            `x` is state variable at the time knots
            `u` is control variable values at time knots
    '''
    t = trajectory['t']
    x = np.array(trajectory['x'], dtype=float)
    u = np.array(trajectory['u'], dtype=float)
    _,nx = x.shape
    _,nu = u.shape

    periodic = np.allclose(x[0], x[-1])
    bc_type = None

    if periodic:
        x[-1] = x[0]
        u[-1] = u[0]
        bc_type = 'periodic'

    xsp = make_interp_spline(t, x, k=5, bc_type=bc_type)
    xsf = bsplinesf(xsp)

    usp = make_interp_spline(t, u, k=3, bc_type=bc_type)
    usf = bsplinesf(usp)

    t,E = construct_basis(t, x, periodic)
    Esp = make_interp_spline(t, E, k=3, bc_type=bc_type)
    Esf = bsplinesf(Esp)

    tau = MX.sym('tau')
    xi = MX.sym('xi', nx-1)
    x = MX.sym('x', nx)
    u = MX.sym('u', nu)
    f = dynamics_rhs(x, u)
    alpha = xsf(tau) + Esf(tau) @ xi
    beta = Esf(tau).T @ (x - xsf(tau))

    J = substitute(jacobian(alpha, xi), xi, 0)
    tmp = jtimes(f, x, J)
    tmp = substitute(tmp, x, xsf(tau))
    Ax = substitute(tmp, u, usf(tau))

    tmp = jacobian(f, u)
    tmp = substitute(tmp, x, xsf(tau))
    Bx = substitute(tmp, u, usf(tau))

    dxsf = jacobian(xsf(tau), tau)
    tmp = jtimes(beta, x, dxsf) + jacobian(beta, tau)
    tmp1 = jtimes(tmp, x, J)
    tmp2 = jtimes(beta, x, Ax)
    tmp = substitute(tmp1 + tmp2, x, xsf(tau))
    tmp = substitute(tmp, u, usf(tau))
    Axi = Function('A', [tau], [tmp])

    tmp = jtimes(beta, x, Bx)
    tmp = substitute(tmp, x, xsf(tau))
    Bxi = Function('B', [tau], [tmp])

    Jsf = Function('J', [tau], [J])

    t = trajectory['t']
    A = np.zeros((len(t), nx-1, nx-1))
    B = np.zeros((len(t), nx-1, nu))
    J = np.zeros((len(t), nx, nx-1))

    for i,w in enumerate(t):
        A[i] = np.array(Axi(w))
        B[i] = np.array(Bxi(w))
        J[i] = np.array(Jsf(w))
    
    if periodic:
        assert np.allclose(A[-1], A[0])
        assert np.allclose(B[-1], B[0])
        assert np.allclose(J[-1], J[0])
        A[-1] = A[0]
        B[-1] = B[0]
        J[-1] = J[0]

    return {
        't': t,
        'A': A,
        'B': B,
        'J': J,
        'E': E
    }
