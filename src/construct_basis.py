import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
from time import time


def solve_ivp_mat(rhs, tspan, X0, **kwargs):
    mshape = np.shape(X0)
    x0 = np.reshape(X0, (-1,))
    vshape = np.shape(x0)

    def vec_rhs(t, x):
        X = np.reshape(x, mshape)
        dX = rhs(t, X)
        dx = np.reshape(dX, vshape)
        return dx
    
    sol = solve_ivp(vec_rhs, tspan, x0, **kwargs)
    x = sol['y'].T
    X = np.reshape(x, (-1,) + mshape)
    t = sol['t']
    return t, X

def tangent(c, t):
    v = c(t, 1)
    n = np.linalg.norm(v, axis=-1)
    if np.shape(t) == ():
        return v / n
    return v / n[:,np.newaxis]

def curvature(c, t):
    '''
        d2gamma/ds2
    '''
    c1 = c(t, 1)
    c2 = c(t, 2)
    c1_sq = np.dot(c1, c1)
    return c2 / c1_sq - c1 * np.dot(c1, c2) / c1_sq**2

def exter(a, b):
    ab = np.outer(a,b)
    return ab - ab.T

def get_perp_vectors(v):
    v = np.reshape(v, (-1,1))
    U,l,Vt = np.linalg.svd(v.T)
    perp = Vt[1:,:] / l[0]
    return perp.T

def get_vec_basis(v):
    v = np.reshape(v, (-1,1))
    U,l,Vt = np.linalg.svd(v.T)
    d = len(v)
    basis = Vt
    if v[:,0].T @ basis[:,0] < 0:
        basis = -basis
    if np.linalg.det(basis) < 0:
        basis[:,-1] = -basis[:,-1]
    return basis

def construct_basis(t, x, periodic=None):
    if periodic is None:
        periodic = np.allclose(x[0], x[-1])
    
    curve = make_interp_spline(t, x, k=3, bc_type='periodic' if periodic else None)
    t1 = t[0]
    t2 = t[-1]
    tan1 = curve(t1, 1)
    tan2 = curve(t2, 1)
    d = len(tan1)

    def A(t):
        cur = curvature(curve, t)
        d = curve(t, 1)
        A = exter(cur, d)
        return A

    def rhs(t,E):
        return A(t).dot(E)

    R0 = get_vec_basis(tan1)
    pt = time()
    t, R = solve_ivp_mat(rhs, [t1, t2], R0, max_step=1e-3, t_eval=t, atol=1e-12, rtol=1e-12, method='RK45')
    pt = time() - pt

    # be sure basis is valid
    tmp = np.transpose(R, (0,2,1)) @ R
    assert np.allclose(tmp, np.eye(d), atol=1e-6)
    tan = tangent(curve, t)
    tmp = tan[:,np.newaxis,:] @ R
    assert np.allclose(tmp[:,:,1:], 0, atol=1e-6)
    assert np.allclose(tmp[:,:,0], 1, atol=1e-6)

    if periodic:
        R = make_basis_periodic(t, R)
        # be sure basis is periodic
        assert np.allclose(R[0], R[-1])
        R[-1] = R[0]

    # first column is the tangent vector, skip it
    basis = R[:,:,1:]
    return t, basis

def make_basis_periodic(t, R):
    t1 = t[0]
    t2 = t[-1]
    R0 = R[0]
    Rn = R[-1]
    log_RnR0 = logm(Rn.T.dot(R0))
    tmp = (t - t1) / (t2 - t1)
    tmp = tmp[:,np.newaxis,np.newaxis] * log_RnR0
    D = np.array([expm(Si) for Si in tmp])
    return R @ D

def main():
    traj = np.load('data/traj.npy', allow_pickle=True).item()
    x = np.concatenate((traj['q'], traj['dq']), axis=1)
    t = traj['t']
    t,E = construct_basis(t, x)
    E_sp = make_interp_spline(t, E, k=3, bc_type='periodic')

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    main()
