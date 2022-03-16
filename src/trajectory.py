import numpy as np


def state_pack(traj : dict):
    t = traj['t']
    ntrailers = traj['ntrailers']
    x = traj['x']
    y = traj['y']
    phi = traj['phi']
    theta = traj['theta']
    st = np.zeros(shape=(len(t), ntrailers + 3))
    st[:,0] = x
    st[:,1] = y
    st[:,2] = phi
    st[:,3:] = theta.T
    return st

def inp_pack(traj : dict):
    t = traj['t']
    ntrailers = traj['ntrailers']
    st = np.zeros(shape=(len(t), ntrailers + 3))
    st[:,0] = traj['u1']
    st[:,1] = traj['u2']
    return st

def state_unpack(st : np.ndarray):
    x = st[:,0] 
    y = st[:,1] 
    phi = st[:,2] 
    theta = st[:,3:].T
    return {
        'x': x,
        'y': y,
        'phi': phi,
        'theta': theta,
    }

def inp_unpack(inp : np.ndarray):
    u1 = inp[:,0] 
    u2 = inp[:,1] 
    return {
        'u1': u1,
        'u2': u2,
    }
