import torch
import numpy as np
from typing import Any

from math import sin, cos, sqrt

from benchmark_function import BenchmarkFunction


class SphereFunction(BenchmarkFunction):
    def __init__(self, d : int, sigma: float = 0, seed: int = 121212, device: str = 'cpu', dtype: torch.dtype = torch.float64) -> None:
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        super().__init__("Sphere", d = d, x_star = x_star, fx_star= 0.0, sigma = sigma, seed = seed, device = device, dtype = dtype)

    def grad(self, x):
        return x

    def __call__(self, x: torch.Tensor) -> float:
        return self.add_noise(0.5 * x.square().sum(dim = 1, keepdim=True))
    
class LeastSquares(BenchmarkFunction):
    def __init__(self, m :int,  d : int, sigma: float = 0, seed: int = 121212, device: str = 'cpu', dtype: torch.dtype = torch.float64, x_star : torch.Tensor | None = None, L : float = None, mu : float=None) -> None:
        x_star = torch.tensor([[i for i in range(d)]], dtype = dtype, device=device).reshape(1, d) if x_star is None else x_star 
        super().__init__("LeastSquares", d = d, x_star = x_star, fx_star= 0.0, sigma = sigma, seed = seed, device = device, dtype = dtype)
        self.m = m
        self.A = torch.randn(self.m, self.d, device=device, dtype=dtype, generator=self.generator)
        if L is not None and mu is not None:
            assert L > mu > 0, "L must be larger than mu and they must be both positive!"
            self.A = self._fix_A(L, mu)
        self.y =  self.A @ x_star.T

    def _fix_A(self, L, mu):
        u, s, vt = torch.svd(self.A.T @ self.A)
        s = torch.linspace(sqrt(mu), sqrt(L), self.d, dtype=self.dtype, device=self.device)
        return u @ torch.diag(s) @ vt
        
    def get_L(self) -> None:
        return torch.linalg.eigvalsh(self.A.T @ self.A)[-1]

    def grad(self, x : torch.Tensor) -> torch.Tensor:
        return  self.A @ ((self.A @ x.T) - self.y)

    def __call__(self, x: torch.Tensor) -> float:
        fx = 0.5 * (torch.linalg.norm((self.A @ x.T) - self.y, dim=1, keepdim=True, ord=2).square())
        return self.add_noise(fx)
    