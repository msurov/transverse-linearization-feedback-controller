import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline


def exterior(a, b):
    ab = np.outer(a, b)
    return ab - ab.T

t_ = np.linspace(0, 1, 10)
v_ = np.random.normal(size=(10, 4))
sp = make_interp_spline(t_, v_, k=3)

def rhs(t, n):
    v = sp(t)
    v1 = sp(t, 1)
    A = -exterior(v, v1) / np.linalg.norm(v)**2
    return A @ n

O1 = np.array([
    [0, 0, 0, -1],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
])
n0 = O1 @ sp(0, 1)
n0 = n0 / np.linalg.norm(n0)

ans = solve_ivp(rhs, [0, 1], n0)
n = ans['y'].T
t = ans['t']
for i in range(len(n)):
    print(np.linalg.norm(n[i]), n[i] @ sp(t[i]))
