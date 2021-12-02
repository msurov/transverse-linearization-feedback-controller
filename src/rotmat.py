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
    assert np.allclose(A @ v, 0)
    return v


def remove_duplicates(_v):
    v = _v[:]
    v.sort()
    duplicates = np.isclose(np.diff(v), 0)
    duplicates = np.append(duplicates,False)
    return v[~duplicates]


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
    comp_eigvals = comp_eigvals[np.imag(comp_eigvals) > 0]
    comp_eigvals = remove_duplicates(comp_eigvals)

    for e in comp_eigvals:
        vecs = nullspace(R - e * I)
        for v in vecs.T:
            v = np.reshape(v, (dim,1))
            v1 = np.real((v + np.conj(v)) / np.sqrt(2))
            v2 = np.real((v - np.conj(v)) / np.complex(0, np.sqrt(2)))
            basis_comp = np.concatenate([basis_comp, v1, v2], axis=1)


    # 4. total basis
    Q = np.concatenate((basis_comp, basis_minus, basis_plus), axis=1)
    if np.isclose(np.linalg.det(Q), -1):
        Q[:,[0,1]] =  Q[:,[1,0]]
    assert np.allclose(np.linalg.det(Q), 1)

    B = Q.T @ R @ Q
    assert np.allclose(R, Q @ B @ Q.T)

    # be sure B is block diagonal
    for i in range(dim//2):
        assert(np.allclose(B[2*i,:2*i], 0)), 'failed'
        assert(np.allclose(B[2*i,2*i+2:], 0)), 'failed'
        assert(np.allclose(B[2*i+1,:2*i], 0)), 'failed'
        assert(np.allclose(B[2*i+1,2*i+2:], 0)), 'failed'

    if dim % 2 == 1:
        assert np.isclose(B[-1,-1], 1)

    return Q,B


def sqrt(R):
    dim = R.shape[0]
    Q,B = rotmat_decompose_QBQt(R)
    sqrt_B = np.eye(dim, dtype=float)

    for i in range(dim//2):
        theta = rotmat2d_angle(B[2*i:2*i+2,2*i:2*i+2])
        sqrt_B[2*i:2*i+2,2*i:2*i+2] = rotmat2d(theta/2)

    return Q @ sqrt_B @ Q.T


def pow(R, k):
    dim = R.shape[0]
    Q,B = rotmat_decompose_QBQt(R)
    sqrt_B = np.eye(dim, dtype=float)

    for i in range(dim//2):
        theta = rotmat2d_angle(B[2*i:2*i+2,2*i:2*i+2])
        sqrt_B[2*i:2*i+2,2*i:2*i+2] = rotmat2d(theta * k)

    return Q @ sqrt_B @ Q.T


def log(R):
    e,V = np.linalg.eig(R)
    if np.any(np.isclose(e, -1)):
        sqrtR = sqrt(R)
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
    return O @ B @ O.T


def test_sqrt():
    for i in range(100):
        np.random.seed(i)
        R = gen_rotmat([np.pi, np.pi/2, 0, 2, 0, np.pi])
        sqrt_R = sqrt(R)
        assert np.allclose(sqrt_R @ sqrt_R, R)

    for i in range(100):
        np.random.seed(i)
        R = gen_rotmat([np.pi, 1,2,3,2,1,2,1,0])
        sqrt_R = sqrt(R)
        assert np.allclose(sqrt_R @ sqrt_R, R)

    for i in range(100):
        np.random.seed(i)
        R = gen_rotmat([np.pi,1,0,3,0,1,2,1,0])
        sqrt_R = sqrt(R)
        assert np.allclose(sqrt_R @ sqrt_R, R)

    for i in range(100):
        np.random.seed(i)
        R = gen_rand_rotmat(11)
        sqrt_R = sqrt(R)
        assert np.allclose(sqrt_R @ sqrt_R, R)


def test_log():
    for i in range(100):
        np.random.seed(i)
        R = gen_rotmat([np.pi, np.pi/2, 0, 2, 0, np.pi])
        sqrt_R = sqrt(R)
        log_sqrt_R = logm(sqrt_R)
        assert np.allclose(R, expm(2*log_sqrt_R))

    for i in range(100):
        np.random.seed(i)
        R = gen_rotmat([1,2,3,2,2,1,2])
        sqrt_R = sqrt(R)
        log_sqrt_R = logm(sqrt_R)
        assert np.allclose(R, expm(2*log_sqrt_R))

    for i in range(100):
        np.random.seed(i)
        R = gen_rand_rotmat(9)
        sqrt_R = sqrt(R)
        log_sqrt_R = logm(sqrt_R)
        assert np.allclose(R, expm(2*log_sqrt_R))

def test_pow():
    for i in range(100):
        np.random.seed(i)
        k, = np.random.random(1)
        R = gen_rand_rotmat(6)
        Rk = pow(R, k)
        Rk2 = expm(k*logm(R))
        assert np.allclose(Rk, Rk2)

    for i in range(100):
        np.random.seed(i)
        k, = np.random.random(1)
        R = gen_rotmat([1,2,3,2,0,2])
        Rk = pow(R, k)
        Rk2 = expm(k*logm(R))
        assert np.allclose(Rk, Rk2)


def test_sqrt2():
    for i in range(100):
        np.random.seed(i)
        R = gen_rotmat([2, 1, np.pi, 2, 0])
        sqrt_R = sqrt(R)
        assert np.allclose(sqrt_R @ sqrt_R, R)

    for i in range(100):
        np.random.seed(i)
        R = gen_rotmat([2, 0, np.pi, 0, 0])
        sqrt_R = sqrt(R)
        assert np.allclose(sqrt_R @ sqrt_R, R)

    for i in range(100):
        np.random.seed(i)
        R = gen_rotmat([2, 0, np.pi, 1, np.pi])
        sqrt_R = sqrt(R)
        assert np.allclose(sqrt_R @ sqrt_R, R)


def test_continuity():
    for i in range(100):
        np.random.seed(i)
        R = gen_rotmat([2, 1, 0, 3, 0, np.pi])
        step = 1e-2
        curve = [pow(R, k) for k in np.arange(0, 1, step)]
        dist = np.array([np.linalg.norm(curve[i] - curve[i+1]) for i in range(0,len(curve)-1)])
        assert np.all(dist < 10 * step)

    for i in range(100):
        np.random.seed(i)
        R = gen_rotmat([2, 1, 2, 3, 3, np.pi])
        step = 1e-2
        curve = [pow(R, k) for k in np.arange(0, 1, step)]
        dist = np.array([np.linalg.norm(curve[i] - curve[i+1]) for i in range(0,len(curve)-1)])
        assert np.all(dist < 10 * step)

    for i in range(100):
        np.random.seed(i)
        R = gen_rand_rotmat(7)
        step = 1e-2
        curve = [pow(R, k) for k in np.arange(0, 1, step)]
        dist = np.array([np.linalg.norm(curve[i] - curve[i+1]) for i in range(0,len(curve)-1)])
        assert np.all(dist < 10 * step)


if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=200)
    test_sqrt()
    test_sqrt2()
    test_log()
    test_pow()
    test_continuity()

