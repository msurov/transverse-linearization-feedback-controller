from casadi import SX, MX, if_else, DM, Function, substitute, reshape
from scipy.interpolate import BSpline
import numpy as np

mm = MX

def Bik(x, i, k, t):
    '''
        x is symbolic variable, it is assumed it belongs to the half-interval [t_{i} .. t_{i+1})
        need to find values of basis functions 
        
            B_{i-k,k}, B_{i-k+1,k}, ..., B_{i,k}
    '''
    D = mm.zeros(k + 1, k + 1)
    D[0,k] = 1

    for r in range(1, k + 1):
        D[r, k-r] = (t[i+1] - x) / (t[i+1] - t[i-r+1]) * D[r-1, k-r+1]
        for a in range(1, r):
            D[r, k-a] = \
                (x - t[i-a]) / (t[i-a+r] - t[i-a]) * D[r-1, k-a] + \
                (t[i-a+r+1] - x) / (t[i-a+r+1] - t[i-a+1]) * D[r-1, k-a+1]
        D[r, k] = (x - t[i]) / (t[i+r] - t[i]) * D[r-1, k]
    
    Bk = D[k,:].T
    return Bk

def select_value(x, t, c):
    '''
        Binary search. Returns symbolic expression which selects c[j] if t[j] <= x < t[j+1]
    '''
    N = len(t)
    if N == 1:
        return c[0]
    return if_else(
        x >= t[N//2],
        select_value(x, t[N//2:], c[N//2:]),
        select_value(x, t[:N//2], c[:N//2]),
    )

def bisect_expr(arg, knots : DM):
    '''
        Get symbolic expression for the bisect algorithm
    '''
    N = knots.shape[0]
    if N == 1:
        return 0
    m = N // 2
    return if_else(
        arg >= knots[m],
        m + bisect_expr(arg, knots[m:]),
        bisect_expr(arg, knots[:m])
    )

def __old__bsplinesf(sp : BSpline, name='BSpline'):
    '''
        Generates CasADi symbolic function for the given B-Spline
        TODO: add extrapolation type arguments
    '''
    t = sp.t
    c = sp.c
    k = sp.k
    N = len(c)
    x = mm.sym('dummy')
    expr = 0
    values = []

    for i in range(k, N):
        val = Bik(x, i, k, t).T @ c[i-k:i+1]
        values += [val]

    expr = select_value(x, t[k:N], values)
    return Function(name, [x], [expr])

def bsplinesf(sp : BSpline, name='BSpline'):
    '''
        Generates CasADi symbolic function for the given B-Spline
        TODO: add extrapolation type arguments
    '''
    t = sp.t
    N,*dshape = sp.c.shape
    if dshape == []:
        dshape = [1,1]
    elif len(dshape) == 1:
        dshape = dshape + [1,]
    total = np.prod(dshape, dtype=int)
    c = np.reshape(sp.c, (N, total))
    k = sp.k
    x = mm.sym('dummy')
    expr = 0
    values = mm.zeros(N-k, total)

    for i in range(k, N):
        values[i - k,:] = Bik(x, i, k, t).T @ c[i-k:i+1,:]

    idx = bisect_expr(x, t[k:N])
    expr = values[idx,:]
    expr = reshape(expr, *dshape[::-1]).T
    return Function(name, [x], [expr])

def bsplinesx(sp : BSpline, arg : mm):
    '''
        Generates CasADi symbolic expression for the given B-Spline
    '''
    f = bsplinesf(sp, name='Dummy')
    return f(arg)

def test1():
    '''
        Visual test
    '''
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline
    import numpy as np

    args = np.linspace(0, 1, 20)
    vals = np.cos(args)
    sp = make_interp_spline(args, vals)
    spsx = bsplinesf(sp)
    spsx1 = spsx.jacobian()

    xx = np.linspace(args[0]-1, args[-1]+1, 100)
    yy = DM([spsx(tmp) for tmp in xx])
    yy1 = DM([spsx1(tmp, 1) for tmp in xx])

    plt.figure('Function')
    plt.plot(xx, yy)
    plt.plot(xx, sp(xx), '--')
    plt.grid(True)

    plt.figure('Derivative')
    plt.plot(xx, yy1)
    plt.plot(xx, sp(xx, 1), '--')
    plt.grid(True)

    plt.show()

def test2():
    '''
        Matrix function interpolation test
    '''
    from scipy.linalg import expm
    from scipy.interpolate import make_interp_spline

    S = np.random.normal(size=(8,8))
    S = S - S.T
    t = np.linspace(0, 1, 371)
    values = np.array([expm(S*w)[:,1:] for w in t])
    sp = make_interp_spline(t, values, k=3)

    spsf = bsplinesf(sp)

    tt = np.linspace(t[0], t[-1], 998)
    for w in tt:
        assert np.allclose(spsf(w), DM(sp(w)))

def test3():
    '''
        Vector function interpolation test
    '''
    from scipy.linalg import expm
    from scipy.interpolate import make_interp_spline

    t = np.linspace(0, 2, 7)
    values = np.array([np.sin(t), np.cos(t), np.exp(t)]).T
    sp = make_interp_spline(t, values, k=3)
    spsf = bsplinesf(sp)

    tt = np.linspace(t[0], t[-1], 1000)
    for w in tt:
        assert np.allclose(spsf(w), DM(sp(w)))

if __name__ == '__main__':
    test1()
    test2()
    test3()
