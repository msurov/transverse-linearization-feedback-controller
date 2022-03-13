import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.interpolate import make_interp_spline
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time


def lqr_lti(A, B, Q, R):
    R'''
        :param A, B: linear system matrices of dimensions nxn, and nxm \
        :param Q, R: weighted matrices of dimensions nxn, mxm \
        :result: K,P
    '''
    P = solve_continuous_are(A, B, Q, R)
    K = -np.linalg.inv(R) @ (B.T @ P)
    return K, P


def lqr_ltv(t, A, B, Q, R, S):
    R'''
        :param t: time knots of length K \
        :param A: array of dim KxNxN, values of matrix A at knots t \
        :param B: array of dim KxNxM, values of matrix B at knots t \
        :param Q: array of dim KxNxN, x weight coefficients \
        :param R: array of dim KxMxM, u weight coefficients \
        :param S: array of dim NxN, x weight coefficients \
        :param method: euler or fine \
        :result:
            K array of dim KxMxN, controller coefficients
            P array of dim KxNxN, solution of the Riccati equation
        Find control of the form u = K x that minimizes the functional
        \[
            J=\int_{0}^{T}u_{s}^{T}R_{s}u_{s}+x_{s}^{T}Q_{s}x_{s}ds+x_{T}^{T}Sx_{T}
        \]
    '''
    N,n,m = B.shape
    assert A.shape == (N,n,n)
    assert Q.shape == (N,n,n)
    assert R.shape == (N,m,m)
    assert S.shape == (n,n)

    fun_A = make_interp_spline(t, A, k=3)
    fun_Q = make_interp_spline(t, Q, k=3)
    inv_R = np.array([np.linalg.inv(R[i,:,:]) for i in range(N)])
    M = np.array([B[i] @ inv_R[i] @ B[i].T for i in range(N)])
    fun_M = make_interp_spline(t, M)

    def rhs(t, p):
        P = np.reshape(p, (n,n))
        A_ = fun_A(t)
        Q_ = fun_Q(t)
        M_ = fun_M(t)
        ATP = A_.T @ P
        dP = -ATP - ATP.T + P @ M_ @ P - Q_
        dp = np.reshape(dP, (-1,))
        return dp

    s = np.reshape(S, (-1,))
    sol = solve_ivp(rhs, [t[-1], t[0]], s, t_eval=t[::-1], max_step=1e-3)
    print(sol.t.shape)
    assert sol.success

    P = np.reshape(sol.y.T, (N, n, n))
    P = P[::-1]

    K = np.zeros((N, m, n))
    for i in range(N):
        K[i] = -inv_R[i] @ B[i].T @ P[i]

    return K, P

def lqr_ltv_periodic(t, A, B, Q, R):
    R'''
        :param t: time knots of length K \
        :param A: array of dim KxNxN, values of matrix A at knots t \
        :param B: array of dim KxNxM, values of matrix B at knots t \
        :param Q: array of dim KxNxN, x weight coefficients \
        :param R: array of dim KxMxM, u weight coefficients \
        :param method: euler or fine \
        :result:
            K array of dim KxMxN, controller coefficients
            P array of dim KxNxN, solution of the Riccati equation
        Find control of the form u = K x that minimizes the functional
        \[
            J=\int_{0}^{T}u_{s}^{T}R_{s}u_{s}+x_{s}^{T}Q_{s}x_{s}ds+x_{T}^{T}Sx_{T}
        \]
    '''
    N,n,m = B.shape
    assert A.shape == (N,n,n)
    assert Q.shape == (N,n,n)
    assert R.shape == (N,m,m)

    fun_A = make_interp_spline(t, A, k=3, bc_type='periodic')
    fun_Q = make_interp_spline(t, Q, k=3, bc_type='periodic')
    inv_R = np.array([np.linalg.inv(R[i,:,:]) for i in range(N)])
    M = np.array([B[i] @ inv_R[i] @ B[i].T for i in range(N)])
    fun_M = make_interp_spline(t, M, bc_type='periodic')

    def rhs(t, p):
        P = np.reshape(p, (n,n))
        A_ = fun_A(t)
        Q_ = fun_Q(t)
        M_ = fun_M(t)
        ATP = A_.T @ P
        dP = -ATP - ATP.T + P @ M_ @ P - Q_
        dp = np.reshape(dP, (-1,))
        return dp

    S = np.zeros((n,n))
    s = np.reshape(S, (-1,))
    i = 0

    while True:
        sol = solve_ivp(rhs, [t[-1], t[0]], s, t_eval=[0], max_step=1e-3)
        if np.allclose(s, sol.y[:,0]):
            break
        s = sol.y[:,0]
        i += 1
        if i > 10:
            raise Exception('Can\'t find periodic solution of MRDE after 10 iterations')

    sol = solve_ivp(rhs, [t[-1], t[0]], s, t_eval=t[::-1], max_step=1e-3)

    P = np.reshape(sol.y.T, (N, n, n))
    P = P[::-1]

    K = np.zeros((N, m, n))
    for i in range(N):
        K[i] = -inv_R[i] @ B[i].T @ P[i]

    return K, P

