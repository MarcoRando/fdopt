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
    
class Quadratic(BenchmarkFunction):
    def __init__(self, d: int, sigma: float = 0, seed: int = 121212, device: str = 'cpu', dtype: torch.dtype = torch.float64, alpha : float = 0.5, beta : float = 0.5) -> None:
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        fx_star = 0.0
        super().__init__('Quadratic', d, x_star, fx_star, sigma, seed, device, dtype)
        self.alpha = alpha
        self.beta = beta
        G = torch.randn((d, d), generator=self.generator, dtype = dtype, device = device)
        self.A = G.T @ G
    
    def __call__(self, x: torch.Tensor) -> float:
        t1 = self.alpha * ((x**4).sum(dtype=self.dtype, dim=1, keepdim=True))
        t2 = torch.zeros((x.shape[0], 1), dtype=self.dtype, device=self.device)
        for i in range(x.shape[0]):
            t2[i] = 0.5 * (x[i, :] @ (self.A @ x[i, :]))
        t3 = self.beta * torch.linalg.norm(x, dim = 1, keepdim=True).square()
        return self.add_noise(t1 + t2 + t3)
    
    
class SparseSphere(BenchmarkFunction):
    def __init__(self, d : int, s : int, sigma: float = 0, seed: int = 121212, device: str = 'cpu', dtype: torch.dtype = torch.float64) -> None:
        assert s < d, "sparsity must be smaller that d"
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        super().__init__("SparseSphere", d = d, x_star = x_star, fx_star= 0.0, sigma = sigma, seed = seed, device = device, dtype = dtype)
        self.used_inds = torch.multinomial(torch.ones(d), replacement=False, num_samples=d - s)

    def grad(self, x):
        g_x = x.clone()
        for i in range(self.d):
            if i not in self.used_inds:
                g_x[:, i] = 0.0
        return g_x

    def __call__(self, x: torch.Tensor) -> float:
        return self.add_noise(0.5 * x[:, self.used_inds].square().sum(dim = 1, keepdim=True))
    
    
class LeastSquares(BenchmarkFunction):
    def __init__(self, m :int,  d : int, sigma: float = 0, seed: int = 121212, device: str = 'cpu', dtype: torch.dtype = torch.float64, x_star : torch.Tensor | None = None, L : float = None, mu : float=None) -> None:
        x_star = torch.tensor([[i for i in range(d)]], dtype = dtype, device=device).reshape(1, d) if x_star is None else x_star 
        super().__init__(f"LeastSquares_{m}", d = d, x_star = x_star, fx_star= 0.0, sigma = sigma, seed = seed, device = device, dtype = dtype)
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
        return  self.A @ ((self.A.mv(x.reshape(self.d))).reshape(-1, 1) - self.y)

    def __call__(self, x: torch.Tensor) -> float:
        fx = torch.zeros(x.shape[0], dtype=self.dtype, device=self.device)
        for i in range(x.shape[0]):
            fx[i] = self.add_noise(0.5 * (torch.linalg.norm((self.A.mv(x[i])).reshape(-1, 1) - self.y, keepdim=True, ord=2).square()))
        return fx.reshape(-1, 1)
    