import numpy as np
from typing import Any

from math import sin, cos

from targets.benchmark_function import BenchmarkFunction


class SphereFunction(BenchmarkFunction):
    def __init__(self, d : int, sigma: float = 0, seed: int = 121212, device: str = 'cpu', dtype: np.dtype = np.float64) -> None:
        x_star = np.asarray([[0.0 for _ in range(d)]], dtype = dtype)
        super().__init__("Sphere", d = d, x_star = x_star, fx_star= 0.0, sigma = sigma, seed = seed, device = device, dtype = dtype)

    def __call__(self, x: np.ndarray, z: Any = None) -> float:
        return self.add_noise(np.square(x).sum(axis=1, dtype=self.dtype))
    
class LeastSquares(BenchmarkFunction):
    def __init__(self, m :int,  d : int, sigma: float = 0, seed: int = 121212, device: str = 'cpu', dtype: np.dtype = np.float64, x_star : np.ndarray | None = None) -> None:
        x_star = np.asarray([[i for i in range(d)]], dtype = dtype) if x_star is None else x_star 
        super().__init__("LeastSquares", d = d, x_star = x_star, fx_star= 0.0, sigma = sigma, seed = seed, device = device, dtype = dtype)
        self.m = m
        self.A = self.rnd_state.randn(self.m, self.d)
        self.y = self.A @ x_star.T
        
    def get_L(self):
        return np.linalg.eigvalsh(self.A.T @ self.A)[-1]

    def __call__(self, x: np.ndarray, z: Any = None) -> float:
        fx = 0.5 * (np.linalg.norm(self.A @ x.T - self.y, ord=2)**2)
        return self.add_noise(fx)
    