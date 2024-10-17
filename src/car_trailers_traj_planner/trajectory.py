import numpy as np
from common.trajectory import Trajectory
from dataclasses import dataclass

@dataclass
class CarTrailersTrajectory(Trajectory):
    @property
    def x(self):
        return self.state[:,0]

    @property
    def y(self):
        return self.state[:,1]

    @property
    def phi(self):
        return self.state[:,2]

    @property
    def theta(self):
        return self.state[:,3:]

    @property
    def u1(self):
        return self.control[:,0]

    @property
    def u2(self):
        return self.control[:,1]

    @property
    def ntrailers(self):
        return np.shape(self.state)[1] - 3

def make_trajectory(
        t : np.ndarray, 
        x : np.ndarray, 
        y : np.ndarray, 
        phi : np.ndarray, 
        theta : np.ndarray, 
        u1 : np.ndarray, 
        u2 : np.ndarray
    ):
    state = np.concatenate((
        [x], [y], [phi], theta
    ), axis=1)
    control = np.concatenate((
        [u1], [u2]
    ), axis=1)
    return CarTrailersTrajectory(
        time=t,
        state=state,
        control=control
    )
