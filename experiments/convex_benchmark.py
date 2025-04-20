import torch
import numpy as np 
from math import sqrt, exp, pi

from synthetic_targets import TargetFunction



class MatyasFunction(TargetFunction):
    
    def __init__(self, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
        super().__init__(2, dtype, device, sigma_noise, seed)
        self.bounds = torch.Tensor([[-10.0, 10.0] for _ in range(2)])
        self.min_f = 0.0
            
    def __call__(self, x):
        return 0.26 * x.norm(p=2, dim=1).square_() - 0.48 * x[:, 0] * x[:, 1]
    
    def grad(self, x):
        grad_x1 = 0.52 * x[:, 0] - 0.48 * x[:, 1]
        grad_x2 = 0.52 * x[:, 1] - 0.48 * x[:, 0]
        
        return torch.stack([grad_x1, grad_x2], dim=1)

class BoothFunction(TargetFunction):
    def __init__(self, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
        super().__init__(2, dtype, device, sigma_noise, seed)
        self.bounds = torch.Tensor([[-10.0, 10.0], [-10.0, 10.0]])
        self.min_f = 0.0
                
    def __call__(self, x):
        return (x[:, 0] + 2 * x[:, 1] - 7.0).square() + (2 * x[:, 0] + x[:, 1] - 5.0).square()
    
    def grad(self, x):
        grad_x1 = 2 * (x[:, 0] + 2 * x[:, 1] - 7) + 4 * (2 * x[:, 0] + x[:, 1] - 5)
        grad_x2 = 4 * (x[:, 0] + 2 * x[:, 1] - 7) + 2 * (2 * x[:, 0] + x[:, 1] - 5)
        
        return torch.stack([grad_x1, grad_x2], dim=1)
    
class Schwefel223Function(TargetFunction):
    
    def __init__(self, d, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
        super().__init__(d, dtype, device, sigma_noise, seed)
        self.min_f = 0.0
        self.bounds = torch.Tensor([[-10.0, 10.0] for _ in range(d)])

    def __call__(self, x):
        return torch.sum(x ** 10, dim=1)

    def grad(self, x):
        return 10 * x ** 9


class ZakharovFunction(TargetFunction):
    def __init__(self, d, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
        super().__init__(d, dtype, device, sigma_noise, seed)
        self.bounds = torch.Tensor([[-5.0, 10.0] for _ in range(d)])
        self.min_f = 0.0
                
    def __call__(self, x):
        n, d = x.shape
        indices = torch.arange(1, d + 1, dtype=x.dtype, device=x.device)  # (d,)
        
        sum_term = 0.5 * (x * indices).sum(dim=1, keepdim=True)  # (n, 1)
        
        result = (x ** 2).sum(dim=1, keepdim=True) + sum_term ** 2 + sum_term ** 4
        return result  # (n, 1)
        
    def grad(self, x):
        n, d = x.shape
        indices = torch.arange(1, d + 1, dtype=x.dtype, device=x.device)  # (d,)
        
        sum_term = 0.5 * (x * indices).sum(dim=1, keepdim=True)  # (n, 1)
        
        grad_x = 2 * x + indices * sum_term + 2 * indices * (sum_term ** 3)
        return grad_x  # (n, d)


# class PowellSum(TargetFunction):
#     def __init__(self, d, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
#         super().__init__(d, dtype, device, sigma_noise, seed)
#         self.min_f = 0.0
#         self.bounds = torch.Tensor([[-1.0, 1.0] for _ in range(d)])

#     def __call__(self, x):
#         indices = torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device)  # Exponents (1+1, 2+1, ..., d+1)
#         return (x.abs().pow(indices + 1)).sum(dim=1)  # Compute sum across dimension d
        
        
#     def grad(self, x):
#         indices = torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device)
#         abs_x = x.abs()
#         grad = (indices + 1) * abs_x.pow(indices - 1) * x.sign() 
#         return grad




# class McCormickFunction(TargetFunction):
#     def __init__(self, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
#         super().__init__(2, dtype, device, sigma_noise, seed)
#         self.bounds = torch.Tensor([[-1.5, 4.0], [-3.0, 3.0]])
#         self.min_f = -1.9132229549810367
                
#     def __call__(self, x):
#         return (x[:, 0] + x[:, 1]).sin() + (x[:, 0] - x[:, 1]).square() - 1.5 * x[:, 0] + 2.5 * x[:,1] + 1
    
#     def grad(self, x):
#         grad_x1 = (x[:, 0] + x[:, 1]).cos() + 2 * (x[:, 0] - x[:, 1]) - 1.5
#         grad_x2 = (x[:, 0] + x[:, 1]).cos() - 2 * (x[:, 0] - x[:, 1]) + 2.5
        
#         return torch.stack([grad_x1, grad_x2], dim=1)

# class BrentFunction(TargetFunction):
#     def __init__(self, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
#         super().__init__(2, dtype, device, sigma_noise, seed)
#         self.bounds = torch.Tensor([[-20.0, 0.0], [-20.0, 0.0]])
#         self.min_f = exp(-200.0)
                
#     def __call__(self, x):
#         return (x[:, 0] + 10.0).square() + (x[:, 1] + 10.0).square() + (-x[:, 0].square() - x[:, 1].square()).exp()
    
#     def grad(self, x):
#         exp_term = (-x[:, 0]**2 - x[:, 1]**2).exp()         
#         grad_x1 = 2 * (x[:, 0] + 10) - 2 * x[:, 0] * exp_term
#         grad_x2 = 2 * (x[:, 1] + 10) - 2 * x[:, 1] * exp_term
        
#         return torch.stack([grad_x1, grad_x2], dim=1)



# class ExponentialFunction(TargetFunction):
#     def __init__(self, d, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
#         super().__init__(d, dtype, device, sigma_noise, seed)
#         self.bounds = torch.Tensor([[-1.0, 1.0] for _ in range(d)])
#         self.min_f = -1.0
                
#     def __call__(self, x):
#         return -(-0.5 * x.square().sum(dim=1)).exp() 
    
#     def grad(self, x):
#         squared_sum = (x**2).sum(dim=1)  # Compute sum(x_i^2) for each row (n, )
#         exp_term = torch.exp(-0.5 * squared_sum)  # Compute the exponential part
#         grad = exp_term.unsqueeze(1) * x  # Multiply by x to get the gradient
        
#         return grad

class TridFunction(TargetFunction):

    def __init__(self, d, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
        super().__init__("Trid", d, dtype, device, sigma_noise, seed)
        self.bounds = torch.tensor([[-d**2, d**2] for _ in range(d)], device=device, dtype=dtype)
        self.min_f = - (d * (d + 4) * (d - 1)) / 6
    
    def __call__(self, x):
        term1 = (x - 1).square().sum(dim=1, keepdim=True)  # Sum of squared terms
        term2 = (x[:, 1:] * x[:, :-1]).sum(dim=1, keepdim=True)  # Sum of product terms (offset by 1)
        
        return term1 - term2  # Compute function

    def grad(self, x):
        n, d = x.shape
        grad = 2 * (x - 1)  # First term: 2(x_i - 1)
        
        # Handle x_1 separately
        grad[:, 0] -= x[:, 1]

        # Handle x_d separately
        grad[:, -1] -= x[:, -2]

        # Handle 2 ≤ i ≤ d-1
        if d > 2:
            grad[:, 1:-1] -= x[:, :-2] + x[:, 2:]

        return grad  # (n, d)