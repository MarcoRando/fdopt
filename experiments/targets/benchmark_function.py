import torch
import numpy as np
from typing import Any

class BenchmarkFunction:
    
    def __init__(self, name : str, d : int, x_star : np.ndarray | None = None, fx_star : float | None = None, sigma : float = 0.0, seed : int = 121212, device : str = 'cpu' , dtype : np.dtype = np.float64) -> None:
        assert x_star is None or x_star.shape[1] == d
        self.name = name
        self.fx_star = fx_star
        self.d = d
        self.sigma = sigma
        self.device = device
        self.dtype = dtype
        self.rnd_state = np.random.RandomState(seed=seed)
        
    def add_noise(self, fx):
        return fx + self.sigma * self.rnd_state.randn(1)
        
        
    def __call__(self, x : np.ndarray,  z : Any = None) -> float:
        pass

    def __str__(self) -> str:
        return "{} [d = {}]".format(self.name, self.d)
    
    def __repr__(self) -> str:
        return "{}_{}".format(self.name, self.d)