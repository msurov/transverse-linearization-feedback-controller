import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import fminbound
import matplotlib.pyplot as plt


class TransverseFeedback:
    R'''
        Trajectory tracking feedback based on the transverse linearization approach
    '''
    def __init__(self, traj : dict, lqr : dict, linsys : dict, periodic=None, window=None):
        R'''
        '''
        self.t = traj.time
        self.t_prev = None

        if window is None:
            self.window = (self.t[-1] - self.t[0]) / 20
        else:
            self.window = window

        self.x_ref = traj.state
        self.u_ref = traj.control

        if periodic is None:
            periodic = np.allclose(self.x_ref[-1], self.x_ref[0])
        
        if periodic:
            self.x_ref[-1] = self.x_ref[0]
            self.u_ref[-1] = self.u_ref[0]

        bc_type = 'periodic' if periodic else None

        self.x_sp = make_interp_spline(self.t, self.x_ref, bc_type=bc_type)
        self.u_sp = make_interp_spline(self.t, self.u_ref, bc_type=bc_type)

        self.K_sp = make_interp_spline(lqr['t'], lqr['K'], bc_type=bc_type)
        self.E_sp = make_interp_spline(linsys['t'], linsys['E'], bc_type=bc_type)

        umin = np.min(self.u_ref, axis=0)
        umax = np.max(self.u_ref, axis=0)
        d = (umax - umin) * 1.2
        umin = umin - d
        umax = umax + d
        self.umin = umin
        self.umax = umax

        self.state = None

    def proj(self, x):
        def f(t):
            d = self.x_sp(t) - x
            return d.T @ d

        if self.t_prev is None:
            d = np.linalg.norm(self.x_ref - x, axis=1)
            i = np.argmin(d)
            t1 = max(self.t[i] - self.window/2, self.t[0])
            t2 = min(self.t[i] + self.window/2, self.t[-1])
            t = fminbound(f, t1, t2, maxfun=100, disp=0)
        else:
            t1 = self.t_prev
            t2 = t1 + self.window
            t = fminbound(f, t1, t2, maxfun=20, disp=0)
        return t

    def get_tau_xi(self, x):
        tau = self.proj(x)
        x_ref = self.x_sp(tau)
        xi = self.E_sp(tau).T @ (x - x_ref)
        return tau, xi

    def process(self, x):
        tau, xi = self.get_tau_xi(x)
        self.t_prev = tau
        self.xi = xi.copy()
        u = self.u_sp(tau) + self.K_sp(tau) @ xi
        u = np.clip(u, self.umin, self.umax)
        self.state = (tau, xi.copy())
        return u

    def __call__(self, x):
        return self.process(x)
