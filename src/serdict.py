from re import L
import numpy as np
from tempfile import gettempdir
from os.path import join

def load(name: str):
    filepath = join(gettempdir(), name)
    return np.load(filepath, allow_pickle=True).item()

def save(name: str, data : dict):
    filepath = join(gettempdir(), name)
    np.save(filepath, data)
