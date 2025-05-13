import torch
import numpy as np 
from math import sqrt, exp, pi

from synthetic_targets import TargetFunction



class StyblinskiTangFunction(TargetFunction):
    
    def __init__(self, d, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
        super().__init__("StyblinskiTang", d, dtype, device, sigma_noise, seed)
        self.bounds = torch.tensor([[-5.0, 5.0] for _ in range(d)], device=device, dtype=dtype)
        self.min_f = -39.16599 * d
        self.x0 = torch.full((1, self.d),0.0,  dtype=self.dtype, device=self.device)
            
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
        self.x0 = torch.full((1, self.d), 0.5, dtype=self.dtype, device=self.device)
            

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
        self.x0 = torch.ones((1, self.d), dtype=self.dtype, device=self.device)
            

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


class DixonPrice(TargetFunction):
    
    def __init__(self, d, dtype=torch.float64, device='cpu', sigma_noise=0.0, seed=121314):
        super().__init__("DixonPrice", d, dtype, device, sigma_noise,seed)
        self.bounds = torch.tensor([[-10.0, 10.0] for _ in range(d)], device=device, dtype=dtype)
        self.min_f = 0.0
        self.x0 = torch.full((1, self.d), 0.0, dtype=self.dtype, device=self.device)   
    
    def __call__(self, x):
        d = x.shape[-1]
        x0 = x[..., 0]
        x_rest = x[..., 1:]    # x1 to xd
        x_prev = x[..., :-1]   # x0 to x_{d-1}

        # Compute the terms
        term1 = (x0 - 1) ** 2
        diff = 2 * x_rest**2 - x_prev
        i = torch.arange(1, d, device=x.device).float()  # 1,2,...,d-1
        term2 = (i * diff**2).sum(dim=-1)

        return self.add_noise(term1 + term2)
    
    def grad(self, x):
        d = x.shape[-1]
        grad = torch.zeros_like(x)
        x0 = x[..., 0]
        x_rest = x[..., 1:]    # x1 to xd
        x_prev = x[..., :-1]   # x0 to x_{d-1}
        i = torch.arange(1, d, device=x.device).float()  # 1,2,...,d-1

        # Compute the terms
        diff = 2 * x_rest**2 - x_prev

        # grad[0]
        grad[..., 0] = 2 * (x0 - 1) - 2 * (i[0] * diff[..., 0])  # From term1 and first term of sum

        # grad[1 to d-1]
        grad[..., 1:] = 8 * i * x_rest * diff

        # grad[1 to d-1] additional terms due to x_{i-1}
        grad[..., :-1] += -2 * i * diff

        return grad