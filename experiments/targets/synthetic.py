import torch

from torch import Tensor, dtype
from typing import Any

from math import sin, cos

from targets.benchmark_function import BenchmarkFunction


class SphereFunction(BenchmarkFunction):
    def __init__(self, d : int, sigma: float = 0, seed: int = 121212, device: str = 'cpu', dtype: dtype = torch.float64) -> None:
        x_star = torch.tensor([[0.0 for _ in range(d)]], dtype = dtype, device = device)
        super().__init__("Sphere", d = d, x_star = x_star, fx_star= 0.0, sigma = sigma, seed = seed, device = device, dtype = dtype)

    def __call__(self, x: Tensor, z: Any = None) -> float:
        return self.add_noise(x.square().sum(dim=1, dtype=self.dtype))
    
class LeastSquares(BenchmarkFunction):
    def __init__(self, m :int,  d : int, sigma: float = 0, seed: int = 121212, device: str = 'cpu', dtype: dtype = torch.float64, x_star : torch.Tensor | None = None) -> None:
        x_star = torch.tensor([[i for i in range(d)]], dtype = dtype, device = device) if x_star is None else x_star 
        super().__init__("LeastSquares", d = d, x_star = x_star, fx_star= 0.0, sigma = sigma, seed = seed, device = device, dtype = dtype)
        self.m = m
        self.A = torch.randn(size=(self.m, self.d), dtype=dtype, device=device, generator=self.generator)
        self.y = self.A @ x_star.T
        
    def get_L(self):
        return torch.linalg.eigvalsh(self.A.T @ self.A)[-1]

    def __call__(self, x: Tensor, z: Any = None) -> float:
        fx = 0.5 * (torch.linalg.norm(self.A @ x.T - self.y, ord=2)**2)
        return self.add_noise(fx)
    