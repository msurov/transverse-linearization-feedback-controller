import numpy as np
import scipy.linalg as la
from scipy.linalg import logm, expm


def gen_rand_rotmat(dim):
    A = np.random.random((dim, dim))
    A = A - A.T
    return expm(A)


def is_real(v, eps):
    return np.all(np.abs(np.imag(v)) < eps)


def is_conj(a, b, eps):
    return np.allclose(a, b.conj(), eps)


def rotmat_sqrt(R, eps=1e-10):
    dim = R.shape[0]
    assert np.abs(np.linalg.det(R) - 1.) < eps
    assert np.allclose(R.dot(R.T), np.eye(dim)), 'Expect argument is a rotation matrix'
    e,V = np.linalg.eig(R)

    C = np.abs(V.T.dot(V) - 1) < eps
    pairs = zip(*np.nonzero(C))
    print pairs
    pairs = filter(lambda (y,x): y <= x, pairs)

    Q = np.zeros((dim,dim), dtype=float)
    B = np.eye(dim, dtype=float)
    i = 0

    for i1,i2 in pairs:
        if i1 == i2:
            Q[:,i] = np.real(V[:,i])
            i++
        else:
            e1 = e[i1]
            e2 = e[i2]
            v1 = V[:,i1]
            v2 = V[:,i2]

            if is_conj(v1, v2, eps):
                assert is_conj(e1, e2, eps)
                Q[:,i] = np.real(v1)
                Q[:,i+1] = np.imag(v1)
                theta = np.angle(e1)
                B[2*i,2*i] = np.cos(theta/2)
                B[2*i,2*i+1] = np.sin(theta/2)
                B[2*i+1,2*i] = -np.sin(theta/2)
                B[2*i+1,2*i+1] = np.cos(theta/2)
            else:
                assert is_real(v1, eps)
                assert is_real(v2, eps)


        e1 = e[2*i]
        e2 = e[2*i+1]
        v1 = V[:,2*i]
        v2 = V[:,2*i+1]

        if abs(e1 + 1) < eps:
            assert abs(e2 + 1.) < eps

            if is_real(v1, eps):
                assert is_real(v2, eps)
                Q[:,2*i] = np.real(v1)
                Q[:,2*i+1] = np.real(v2)
                B[2*i,2*i] = 0
                B[2*i,2*i+1] = 1
                B[2*i+1,2*i] = -1
                B[2*i+1,2*i+1] = 0
            else:
                assert is_conj(v1,v2,eps)
                Q[:,2*i] = np.real(v1)
                Q[:,2*i+1] = np.imag(v1)
                B[2*i,2*i] = 0
                B[2*i,2*i+1] = 1
                B[2*i+1,2*i] = -1
                B[2*i+1,2*i+1] = 0

        elif abs(e1 - 1.) < eps:
            assert abs(e2 - 1.) < eps
            if is_real(v1, eps) and is_real(v2, eps):
                Q[:,2*i] = np.real(v1)
                Q[:,2*i+1] = np.real(v2)
            else:
                print v1
                print v2
                assert is_conj(v1, v2, eps)
                Q[:,2*i] = np.real(v1)
                Q[:,2*i+1] = np.imag(v1)
        else:
            assert abs(e1 * e2 - 1.) < eps
            assert is_conj(v1,v2,eps)

            Q[:,2*i] = np.real(v1)
            Q[:,2*i+1] = np.imag(v1)
            theta = np.angle(e1)
            B[2*i,2*i] = np.cos(theta/2)
            B[2*i,2*i+1] = np.sin(theta/2)
            B[2*i+1,2*i] = -np.sin(theta/2)
            B[2*i+1,2*i+1] = np.cos(theta/2)

    if dim % 2 == 1:
        e1 = e[-1]
        v1 = V[:,-1]
        assert abs(e1 - 1.) < eps
        assert np.allclose(np.imag(v1), 0.)
        Q[:,-1] = np.real(v1)
        B[-1,-1] = 1.

    Qi = np.linalg.inv(Q)
    return Q.dot(B).dot(Qi)


def gen_rotmat(angles):
    dim = 2 * len(angles)
    B = np.eye(dim)

    for i,a in enumerate(angles):
        B[2*i,2*i] = np.cos(a)
        B[2*i,2*i+1] = -np.sin(a)
        B[2*i+1,2*i] = np.sin(a)
        B[2*i+1,2*i+1] = np.cos(a)

    O = gen_rand_rotmat(dim)
    return O.dot(B).dot(O.T)


def rotmat_log(R):
    pass

np.set_printoptions(suppress=True, linewidth=200)

for i in range(1000):
    np.random.seed(i)
    R = gen_rotmat([0., np.pi, 1, np.pi/2, 0.])
    sqrtR = rotmat_sqrt(R)
    assert np.allclose(sqrtR.dot(sqrtR), R)
