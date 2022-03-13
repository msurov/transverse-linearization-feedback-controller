import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import fminbound


class TransversePeriodicFeedback:
    R'''
        Trajectory tracking feedback based on the transverse linearization approach
    '''
    def __init__(self, traj : dict, fb : dict, linsys : dict):
        R'''
            Reference trajectory \
                `traj` is reference trajectory \
                `coef` is a solution of LTV LQR problem
        '''
        self.t = traj['t']
        self.t_prev = None

        self.window = 0.2
        self.x_ref = traj['x']
        self.u_ref = traj['u']

        self.x_sp = make_interp_spline(self.t, self.x_ref, bc_type='periodic')
        self.u_sp = make_interp_spline(self.t, self.u_ref, bc_type='periodic')

        self.K_sp = make_interp_spline(fb['t'], fb['K'], bc_type='periodic')
        self.E_sp = make_interp_spline(linsys['t'], linsys['E'], bc_type='periodic')

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
