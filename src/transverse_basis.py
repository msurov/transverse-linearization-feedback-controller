import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from misc.math.math import integrate
from scipy.linalg import expm, logm
from time import time


class Curve:
    def __init__(self, dim):
        self.nfreq = 3
        self.ndim = dim

        self.a = np.random.random((self.nfreq, self.ndim)) - 0.5
        self.b0 = np.random.random(self.ndim) - 0.5
        self.b = np.random.random((self.nfreq, self.ndim)) - 0.5

        self.n = np.arange(1, self.nfreq+1)
        self.an = np.reshape(self.n, (-1,1)) * self.a
        self.bn = np.reshape(self.n, (-1,1)) * self.b
        self.an2 = np.reshape(self.n**2, (-1,1)) * self.a
        self.bn2 = np.reshape(self.n**2, (-1,1)) * self.b


    def val(self, t):
        sin_nt = np.sin(np.outer(t, self.n))
        cos_nt = np.cos(np.outer(t, self.n))
        return sin_nt.dot(self.a) + cos_nt.dot(self.b) + self.b0


    def der1(self, t):
        '''
            d gamma / dt
        '''
        sin_nt = np.sin(np.outer(t, self.n))
        cos_nt = np.cos(np.outer(t, self.n))
        return cos_nt.dot(self.an) - sin_nt.dot(self.bn)


    def der2(self, t):
        '''
            d2 gamma / dt2
        '''
        sin_nt = np.sin(np.outer(t, self.n))
        cos_nt = np.cos(np.outer(t, self.n))
        return -sin_nt.dot(self.an2) - cos_nt.dot(self.bn2)


    def tangent(self, t):
        '''
            d gamma / ds
        '''
        d = self.der1(t)
        k = 1. / np.linalg.norm(d, axis=1)
        k = np.reshape(k, (len(d), 1))
        return k * d


    def curvature(self, t):
        '''
            d2gamma/ds2
        '''
        d1 = self.der1(t)
        d2 = self.der2(t)
        k = 1. / np.linalg.norm(d1, axis=1)
        k = np.reshape(k, (len(d1), 1))
        d1d2 = np.reshape(np.sum(d1 * d2, axis=1), (-1,1))
        return k**2 * d2 - k**4 * d1 * d1d2


    def speed(self, t):
        '''
            ds/dt
        '''
        d = self.der1(t)
        v = np.linalg.norm(d, axis=1)
        v = np.reshape(v, (len(d), 1))
        return v


    def __call__(self, t):
        return self.val(t)



def get_perp(v):
    v = np.reshape(v, (-1,1))
    U,l,Vt = np.linalg.svd(v.T)
    perp = Vt[1:,:] / l[0]
    return perp.T


def get_basis(v):
    v = np.reshape(v, (-1,1))
    U,l,Vt = np.linalg.svd(v.T)
    d = len(v)
    basis = Vt / l[0]
    if np.linalg.det(basis) < 0:
        basis[:,-1] = -basis[:,-1]
    return basis


def test():
    np.random.seed(3)

    T = 2*np.pi
    t = np.linspace(0, T, 10000)
    ndim = 3
    gamma = Curve(ndim)


    def A(t):
        cur, = gamma.curvature(t)
        d, = gamma.der1(t)
        A = exter(cur, d)
        return A

    def rhs(t,E):
        return A(t).dot(E)


    tau0, = gamma.tangent(0.)
    E0 = get_perp(tau0)
    t, E = integrate(rhs, E0, [0, T], step=1e-3)
    assert np.all(np.abs([gamma.tangent(ti).dot(Ei) for ti,Ei in zip(t,E)]) < 1e-10), 'E is not perpendicular to tangent'
    assert np.all([np.allclose(np.dot(Ei.T, Ei), np.eye(ndim-1)) for Ei in E]), 'E is not orthonormal'


def exter(a, b):
    ab = np.outer(a,b)
    return ab - ab.T


def test2():
    np.random.seed(3)

    T = 2*np.pi
    t = np.linspace(0, T, 1000)
    ndim = 5
    gamma = Curve(ndim)

    def A(t):
        cur, = gamma.curvature(t)
        d, = gamma.der1(t)
        A = exter(cur, d)
        return A

    def rhs(t,E):
        return A(t).dot(E)

    tau0, = gamma.tangent(0.)
    E0 = get_perp(tau0)
    t, E = integrate(rhs, E0, [0, T], step=1e-3)
    En = E[-1]
    # print En.T.dot(E0)
    # print E0.dot(E0.T)


    log_EnE0 = logm(En.T.dot(E0))
    F = [Ei.dot(expm(ti / T * log_EnE0)) for ti,Ei in zip(t,E)]

    assert np.all(np.abs([gamma.tangent(ti).dot(Fi) for ti,Fi in zip(t,F)]) < 1e-10), 'F is not perpendicular to tangent'
    assert np.all([np.allclose(np.dot(Fi.T, Fi), np.eye(ndim-1)) for Fi in F]), 'F is not orthonormal'


def test3():
    np.random.seed(3)

    T = 2*np.pi
    t = np.linspace(0, T, 1000)
    ndim = 10
    gamma = Curve(ndim)


    def A(t):
        cur, = gamma.curvature(t)
        d, = gamma.der1(t)
        A = exter(cur, d)
        return A

    def rhs(t,E):
        return A(t).dot(E)


    tau0, = gamma.tangent(0.)
    R0 = get_basis(tau0)
    t, R = integrate(rhs, R0, [0, T], step=1e-3)
    Rn = R[-1]

    assert np.all([abs(np.linalg.det(Ri) - 1) < 1e-6 for Ri in R]), 'R is not a rotation matrix'
    assert np.all([np.allclose(Ri.T.dot(Ri), np.eye(ndim)) for Ri in R]), 'R is not a rotation matrix'

    log_RnR0 = logm(Rn.T.dot(R0))
    F = [Ri.dot(expm(ti / T * log_RnR0)) for ti,Ri in zip(t,R)]

    assert np.all([abs(np.linalg.det(Fi) - 1) < 1e-6 for Fi in F]), 'F is not a rotation matrix'
    assert np.all([np.allclose(Fi.T.dot(Fi), np.eye(ndim)) for Fi in F]), 'F is not a rotation matrix'
    tmp = np.zeros((1,ndim), dtype=float)
    tmp[0,0] = 1
    assert np.all([np.allclose(gamma.tangent(ti).dot(Fi), tmp) for ti,Fi in zip(t,F)]), 'F is not perp to tau'

    F0 = F[0]
    Fn = F[-1]
    assert np.allclose(F0.T.dot(Fn), np.eye(ndim)), 'F is not periodic'


def plot_vector_field(pts, vecs, **kwargs):
    assert len(pts) == len(vecs), 'Expect pts, vecs are of the same lengths, but pts is of %d, vecs is of %d' % (len(pts), len(vecs))
    for p,v in zip(pts, vecs):
        plt.quiver(p[0], p[1], p[2], v[0], v[1], v[2], pivot='tail', **kwargs)


def test_vis():
    np.random.seed(2)

    T = 2*np.pi
    t = np.linspace(0, T, 1000)
    ndim = 3
    gamma = Curve(ndim)

    def A(t):
        cur, = gamma.curvature(t)
        d, = gamma.der1(t)
        A = exter(cur, d)
        return A

    def rhs(t,E):
        return A(t).dot(E)


    tau0, = gamma.tangent(0.)
    R0 = get_basis(tau0)
    t, R = integrate(rhs, R0, [0, T], step=1e-3)
    Rn = R[-1]

    log_RnR0 = logm(Rn.T.dot(R0))
    F = np.array([Ri.dot(expm(ti / T * log_RnR0)) for ti,Ri in zip(t,R)])

    fig = plt.figure('basis')
    r = gamma(t)
    step = 50
    pts = [gamma(ti)[0] for ti in t[::step]]

    # basis R
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(r[:,0], r[:,1], r[:,2], linewidth=5.)
    ax.plot(r[0:1,0], r[0:1,1], r[0:1,2], 'o')

    e1 = [Ri[:,1] / 10 for Ri in R[::step]]
    e2 = [Ri[:,2] / 10 for Ri in R[::step]]
    plot_vector_field(pts, e1, color='blue', linewidth=0.5)
    plot_vector_field(pts, e2, color='blue', linewidth=0.5)

    # basis F
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot(r[:,0], r[:,1], r[:,2], linewidth=5.)
    ax.plot(r[0:1,0], r[0:1,1], r[0:1,2], 'o')

    e1 = [Fi[:,1] / 10 for Fi in F[::step]]
    e2 = [Fi[:,2] / 10 for Fi in F[::step]]
    plot_vector_field(pts, e1, color='green', linewidth=0.5)
    plot_vector_field(pts, e2, color='green', linewidth=0.5)

    plt.show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=200)
    test()
    test2()
    test_vis()
