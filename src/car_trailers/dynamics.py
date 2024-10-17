import casadi as ca
import numpy as np


def product(f, i1, i2):
    R'''
        find product of function `f` of integer arguments from `i1` to `i2`:
        `f(i1) * f(i1+1) * ... * f(i2)`
    '''
    P = 1
    for i in range(i1, i2+1):
        P = P * f(i)
    return P


def sum(f, i1, i2):
    R'''
        find sum of function `f` of integer arguments from `i1` to `i2`:
        `f(i1) + f(i1+1) + ... + f(i2)`
    '''
    S = 0
    for i in range(i1, i2+1):
        S = S + f(i)
    return S


def car_trailers_dynamics(ntrailers : int):
    R'''
        The first car has 4 wheels;
        `phi` is the angle of the steering wheels (wrt x-axis of the first wheel) \
        `thetaI` is the orientation of I-th trailer \
        `x,y` are the cartesian coordinates of the centre of car's forward wheels \
    '''
    nstates = 3 + ntrailers
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    phi = ca.SX.sym('phi')
    thetas = ca.SX.sym('theta', ntrailers)
    u = ca.SX.sym('u', 2)

    ntrailers = ntrailers
    state = ca.vertcat(x, y, phi, thetas)

    g1 = ca.SX.zeros((nstates, 1))
    g1[0] = ca.cos(thetas[0])
    g1[1] = ca.sin(thetas[0])

    if ntrailers > 0:
        g1[3] = 1 * ca.tan(phi)

    for i in range(1, ntrailers):
        fun = lambda j: ca.cos(thetas[j-1] - thetas[j])
        g1[3+i] = product(fun, 1, i-1) * ca.sin(thetas[i-1] - thetas[i])
    
    g2 = ca.SX.zeros((nstates, 1))
    g2[2] = 1

    g = ca.horzcat(g1, g2)
    f = ca.SX.zeros((nstates, 1))
    rhs_expr = f + g @ u
    rhs_fun = ca.Function('CarTrailersRHS', [state, u], [rhs_expr])
    return rhs_fun

'''
def trailer_position(i):
    assert i >= 0 and i < self.ntrailers
    x = self.x - sum(lambda j: ca.cos(self.thetas[j]), 1, i)
    y = self.y - sum(lambda j: ca.sin(self.thetas[j]), 1, i)
    return x,y
'''
