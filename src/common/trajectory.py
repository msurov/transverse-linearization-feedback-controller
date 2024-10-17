from dataclasses import dataclass
import numpy as np


@dataclass 
class Trajectory:
	time : np.ndarray
	state : np.ndarray
	control : np.ndarray

	def __post_init__(self):
		self.time = np.array(self.time, dtype=float)
		self.state = np.array(self.state, dtype=float)
		self.control = np.array(self.control, dtype=float)
		nt, = self.time.shape
		assert self.state.shape[0] == nt
		assert self.control.shape[0] == nt
