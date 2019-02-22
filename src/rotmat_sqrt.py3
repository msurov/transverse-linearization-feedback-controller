import numpy as np
from scipy.linalg import logm, expm


def is_real(v, eps):
    return np.all(np.abs(np.imag(v)) < eps)


def is_conj(a, b, eps):
    return np.allclose(a, b.conj(), eps)


def rotmat2d(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])


def rotmat2d_angle(R):
    c,s = R[:,0]
    return np.arctan2(s,c)


def nullspace(A, eps=1e-10):
    U,s,Vt = np.linalg.svd(A)
    m = s < eps
    i, = np.nonzero(m)
    v = np.conj(Vt[i,:]).T
    assert np.allclose(A.dot(v), 0)
    return v


def rotmat_decompose_QBQt(R):
    dim = R.shape[0]
    I = np.eye(dim, dtype=float)
    eigvals,_ = np.linalg.eig(R)

    # real and complex eigenvalues
    m = np.isclose(np.imag(eigvals), 0)
    real_idx, = np.nonzero(m)
    real_eigvals = np.real(eigvals[real_idx])
    comp_idx, = np.nonzero(~m)
    comp_eigvals = eigvals[comp_idx]
    comp_eigvals = comp_eigvals[np.imag(comp_eigvals) > 0]

    # 1. basis of eigval 1
    basis_plus = np.zeros((dim,0))
    if np.any(np.isclose(real_eigvals, 1)):
        basis_plus = nullspace(R - I)

    # 2. basis of eigval -1
    basis_minus = np.zeros((dim,0))
    if np.any(np.isclose(real_eigvals, -1)):
        basis_minus = nullspace(R + I)

    # 3. basis of conj eigvals
    basis_comp = np.zeros((dim,0))

    for e in comp_eigvals:
        vecs = nullspace(R - e * I)
        vecs1 = np.real((vecs + np.conj(vecs)) / np.sqrt(2))
        vecs2 = np.real((vecs - np.conj(vecs)) / np.complex(0, np.sqrt(2)))
        basis_comp = np.concatenate((basis_comp, vecs1, vecs2), axis=1)

    # 4. total basis
    Q = np.concatenate((basis_comp, basis_minus, basis_plus), axis=1)
    B = Q.T.dot(R).dot(Q)

    # be sure B is block diagonal
    for i in range(dim//2):
        assert(np.allclose(B[2*i,:2*i], 0)), 'failed'
        assert(np.allclose(B[2*i,2*i+2:], 0)), 'failed'
        assert(np.allclose(B[2*i+1,:2*i], 0)), 'failed'
        assert(np.allclose(B[2*i+1,2*i+2:], 0)), 'failed'

    if dim % 2 == 1:
        assert np.isclose(B[-1,-1], 1)

    assert np.allclose(R, Q.dot(B).dot(Q.T))
    assert np.allclose(np.linalg.det(Q), 1)

    return Q,B


def rotmat_sqrt(R):
    dim = R.shape[0]
    Q,B = rotmat_decompose_QBQt(R)
    sqrt_B = np.eye(dim, dtype=float)

    for i in range(dim//2):
        theta = rotmat2d_angle(B[2*i:2*i+2,2*i:2*i+2])
        sqrt_B[2*i:2*i+2,2*i:2*i+2] = rotmat2d(theta/2)

    return Q.dot(sqrt_B).dot(Q.T)


def rotmat_pow(R, k):
    dim = R.shape[0]
    Q,B = rotmat_decompose_QBQt(R)
    sqrt_B = np.eye(dim, dtype=float)

    for i in range(dim//2):
        theta = rotmat2d_angle(B[2*i:2*i+2,2*i:2*i+2])
        sqrt_B[2*i:2*i+2,2*i:2*i+2] = rotmat2d(theta * k)

    return Q.dot(sqrt_B).dot(Q.T)


def rotmat_logm(R):
    e,V = np.linalg.eig(R)
    if np.any(np.isclose(e, -1)):
        sqrtR = rotmat_sqrt(R)
        e,V = np.linalg.eig(sqrtR)
        halfA = logm(sqrtR)
        return 2*halfA
    return logm(R)


def gen_rand_rotmat(dim):
    A = np.random.random((dim, dim))
    A = A - A.T
    return expm(A)


def gen_rotmat(angles):
    dim = 2 * len(angles)
    B = np.eye(dim)

    for i,a in enumerate(angles):
        B[2*i:2*i+2,2*i:2*i+2] = rotmat2d(a)

    O = gen_rand_rotmat(dim)
    return O.dot(B).dot(O.T)


def test_rotmat_sqrt():
    for i in range(1000):
        np.random.seed(i)
        R = gen_rotmat([np.pi, np.pi/2, 0, 2, 0, np.pi])
        # R = gen_rand_rotmat(7)
        sqrt_R = rotmat_sqrt(R)
        assert np.allclose(sqrt_R.dot(sqrt_R) - R, 0)


def test_logm():
    for i in range(1000):
        np.random.seed(i)
        R = gen_rotmat([np.pi, np.pi/2, 0, 2, 0, np.pi])
        sqrt_R = rotmat_sqrt(R)
        log_sqrt_R = logm(sqrt_R)
        assert np.allclose(R - expm(2*log_sqrt_R), 0)


def test_pow():
    np.random.seed(0)
    k = 0.1254236
    R = gen_rand_rotmat(6)
    Rk = rotmat_pow(R, k)
    Rk2 = expm(k*logm(R))
    np.allclose(Rk, Rk2)


def test_rotmat_sqrt2():
    np.random.seed(0)
    R = gen_rotmat([2, 0, 2, 1])
    sqrt_R = rotmat_sqrt(R)
    assert np.allclose(sqrt_R.dot(sqrt_R) - R, 0)


if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=200)
    test_rotmat_sqrt2()
