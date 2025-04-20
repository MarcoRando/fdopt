import torch
import numpy as np 
from math import sqrt, exp, pi

from synthetic_targets import TargetFunction



class StyblinskiTangFunction(TargetFunction):
    
    def __init__(self, d, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
        super().__init__("StyblinskiTang", d, dtype, device, sigma_noise, seed)
        self.bounds = torch.tensor([[-5.0, 5.0] for _ in range(d)], device=device, dtype=dtype)
        self.min_f = -39.16599 * d
            
    def __call__(self, x):
        x = x if x.ndim > 1 else x.unsqueeze(0)
        fx = 0.5 * (x**4 - 16 * x**2 + 5 * x).sum(dim=1)
        return fx.squeeze()
    
    def grad(self, x):
        x = x if x.ndim > 1 else x.unsqueeze(0)
        grad = 2 * x**3 - 16 * x + 2.5
        return grad.squeeze()

class RosenbrockFunction(TargetFunction):
    def __init__(self, d, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
        super().__init__("Rosenbrock", d, dtype, device, sigma_noise, seed)
        self.bounds = torch.tensor([[-5.0, 10.0] for _ in range(d)], device=device, dtype=dtype)
        self.min_f = 0.0
            

    def __call__(self, x):
        x = x if x.ndim > 1 else x.unsqueeze(0)  # (batch_size, d)
        xi = x[:, :-1]
        xi1 = x[:, 1:]
        return (100 * (xi1 - xi**2)**2 + (1 - xi)**2).sum(dim=1).squeeze()

    def grad(self, x):
        x = x if x.ndim > 1 else x.unsqueeze(0)  # (batch_size, d)
        batch_size, d = x.shape
        grad = torch.zeros_like(x)

        # Compute intermediate terms
        xi = x[:, :-1]
        xi1 = x[:, 1:]
        grad[:, :-1] += -400 * xi * (xi1 - xi**2) - 2 * (1 - xi)
        grad[:, 1:] += 200 * (xi1 - xi**2)

        return grad.squeeze()


class QuingFunction(TargetFunction):
    def __init__(self, d, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
        super().__init__("Quing", d, dtype, device, sigma_noise, seed)
        self.bounds = torch.tensor([[-500.0, 500.0] for _ in range(d)], device=device, dtype=dtype)
        self.min_f = 0.0
            

    def __call__(self, x):
        x = x if x.ndim > 1 else x.unsqueeze(0)
        d = x.shape[1]
        i = torch.arange(1, d + 1, device=x.device).unsqueeze(0)  # shape (1, d)
        return ((x ** 2 - i) ** 2).sum(dim=1).squeeze()

    def grad(self, x):
        x = x if x.ndim > 1 else x.unsqueeze(0)
        d = x.shape[1]
        i = torch.arange(1, d + 1, device=x.device).unsqueeze(0)  # shape (1, d)
        grad = 4 * x * (x ** 2 - i)
        return grad.squeeze()
