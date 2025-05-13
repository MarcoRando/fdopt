import torch
import numpy as np 
import pycutest
from math import sqrt

import pandas as pd

from sklearn.datasets import load_svmlight_file


def load_libsvm_data(datapath, device = 'cpu', dtype = torch.float32):
    X, y = load_svmlight_file(datapath)
    X = torch.from_numpy(X.toarray()).to(dtype=dtype,device=device)
    y = torch.tensor(y, device=device, dtype=dtype)
    return X, y


class TargetFunction:
    
    def __init__(self, name, d, dtype = torch.float64, device = 'cpu', sigma_noise = 0.0, seed = 121314) -> None:
        self.name = name
        self.d = d
        self.dtype = dtype
        self.device = device
        self.sigma_noise = sigma_noise
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def add_noise(self, fx):
        eps = self.sigma_noise * torch.randn(fx.shape[0], generator=self.generator, device=self.device, dtype=self.dtype)
        return fx + eps

        
    def __call__(self, x):
        pass
        
    def grad(self, x):
        pass
        
    def close(self):
        pass
        
    def __str__(self):
        return f"{self.name}_{self.d}"

class LeastSquares(TargetFunction):
    
    def __init__(self, d, mu, L, set_l = True, dtype = torch.float64, device = 'cpu', sigma_noise = 0.0, seed = 121314) -> None:
        assert mu < L
        super().__init__(name="LeastSquares", d = d, dtype = dtype, device= device, sigma_noise=sigma_noise, seed=seed)
        self.mu = mu
        self.L = L
        self.bounds = torch.tensor([[-1, 1] for _ in range(d)], device=device, dtype=dtype)
        A = torch.randn(self.d, self.d, generator=self.generator, dtype=self.dtype, device=self.device)
        if set_l:
            Q, _ = torch.linalg.qr(A)
            S = torch.diag(torch.linspace(sqrt(L), sqrt(mu), self.d, dtype=self.dtype, device=self.device))
            self.A = Q @ S @ Q.T
        else:
            self.A = A.T @ A
        self.x_star = torch.randn(1, self.d, generator=self.generator, device=self.device, dtype=self.dtype) 
        self.x0 = torch.ones((1, self.d), dtype=self.dtype, device=self.device)
        self.y = self.A @ self.x_star.T
        self.min_f = 0.0
        
    def __call__(self, x):
        fx = torch.linalg.norm(self.A @ x.T - self.y, dim=0).square().mul(0.5)
        return self.add_noise(fx)

    def grad(self, x):
        return self.A.T @ (self.A @ x.T - self.y)

    def __str__(self):
        return f"{self.name}_{self.L}_{self.mu}_{self.d}"
    
class SparseSum(TargetFunction):
    
    def __init__(self, d, s, dtype = torch.float64, device = 'cpu', sigma_noise = 0.0, seed = 121314) -> None:
        super().__init__(name=f"SparseSum_{s}", d = d, dtype = dtype, device= device, sigma_noise=sigma_noise, seed=seed)
        self.s = s
        self.inds = torch.randperm(self.d, device=self.device)[:self.s]
        self.x0 = torch.full((1, self.d), 1.0, dtype=self.dtype, device=self.device)
        self.y = torch.randn(self.s, dtype=self.dtype, device=self.device)
        self.bounds = torch.tensor([[-1, 1] for _ in range(d)], device=device, dtype=dtype)
        self.min_f = 0.0
        
    def __call__(self, x):
        return self.add_noise(0.5 * torch.sum((x[:, self.inds] - self.y) **2, dim =1))

    def grad(self, x):
        grad = torch.zeros_like(x)
        grad[0, self.inds] = (x[0, self.inds] - self.y)
        return grad

class Quadratic(TargetFunction):

    def __init__(self, d, dtype=torch.float64, device='cpu', seed=121314):
        super().__init__("Quadratic", d, dtype, device, seed)
        self.x0 = torch.ones((1, self.d), dtype=self.dtype, device=self.device)
        self.b = torch.randn(self.d, dtype=self.dtype, device=self.device)
        self.bounds = torch.tensor([[-1, 1] for _ in range(d)], device=device, dtype=dtype)
        self.min_f = 0.0
        self.A = torch.logspace(-3, 3, self.d, dtype=self.dtype, device=self.device)

    def __call__(self, x):
        Ax = self.A * x
        quadratic = 0.5 * (x * Ax).sum(dim=-1) 
        linear_term = (self.b * x).sum(dim=-1)
        return self.add_noise(quadratic - linear_term)

    def grad(self, x):
        return self.A * x - self.b 


class RegularizedLogistic(TargetFunction):
    
    def __init__(self, d, n, lam = 1e-5, dtype=torch.float64, device='cpu', sigma_noise=0, seed=121314):
        super().__init__("LogisticReg", d, dtype, device, sigma_noise, seed)
        self.X = torch.randn((n, d), dtype=dtype, device=device, generator=self.generator) 
        self.w_star = torch.randn((d,), generator=self.generator, dtype=dtype, device=device)
        self.y = (self.X @ self.w_star).sign()
        self.bounds = torch.tensor([[-1, 1] for _ in range(d)], device=device, dtype=dtype)
                
        self.min_f = 0.0
        self.x0 = torch.full((1, self.d), -1.0, dtype=self.dtype, device=self.device)
        
        self.lam = lam
    
    def __call__(self, w):
        expn = (1 + (-torch.matmul(w, self.X.T) * self.y).exp_()).log_().mean(dim=1)
        return self.add_noise(expn + self.lam * w.norm(p=2, dim=1).square())
    
    def grad(self, w):
        num = self.X * -self.y.view(-1, 1) # n x d
        den = (1 + (torch.matmul(w, self.X.T) * self.y).exp_()).view(w.shape[0], self.X.shape[0], 1) # l x n
        t1 =  (num/den).mean(1)
        t2 = 2 * self.lam * w
        
        return t1 + t2
    def __str__(self):
        return f"{self.name}_{self.X.shape[0]}_{self.d}"
    
class SyntheticTarget(TargetFunction):

    def __init__(self, d, mu, L, dtype = torch.float64, device = 'cpu', sigma_noise = 0.0, seed = 121314) -> None:
        assert mu < L
        super().__init__(d = d, dtype = dtype, device= device, sigma_noise=sigma_noise, seed=seed)
        self.mu = mu
        self.L = L
        self.x0 = torch.zeros((1, self.d), dtype=self.dtype, device=self.device)
        self.ones_matrix = torch.ones((self.d, self.d), dtype=dtype, device=device)

        
    def __call__(self, x):
        n = self.d
        assert n % 2 == 0, "n must be even"        

        indices_odd = torch.arange(0, n, 2, device=x.device)
        indices_even = torch.arange(1, n, 2, device=x.device)
        term1 = self.mu * torch.sin(x[..., indices_odd]) + torch.cos(x[..., indices_even])
        term1_sum = term1.sum(dim=-1)
#        print(term1_sum)
        # term2_values = torch.zeros((x.shape[0],), dtype=x.dtype, device=x.device)
        # for i in range(x.shape[0]):
        #     term2_values[i] = (self.L - self.mu) / (2 * n) * (x[i, :] @ (self.ones_matrix @ x[i, :].T))
        term2_values = (self.L - self.mu) / (2 * n) * ( x @ self.ones_matrix.T * x).sum(dim=-1)  #(x.reshape([i, :] @ (self.ones_matrix @ x[i, :].T))    
        
#        print(term2_values)
#        print(t2)
        
#        quadratic_term = (self.L - self.mu) / (2 * n) * (x @ self.ones_matrix @ x.transpose(-1, -2))
        fx = self.add_noise(term1_sum + term2_values)
#        print(fx.shape)
        return fx.flatten()

    def grad(self, x):
        n = self.d
        assert n % 2 == 0, "n must be even"        
        grad = torch.zeros_like(x)
        indices_odd = torch.arange(0, n, 2, device=x.device)
        indices_even = torch.arange(1, n, 2, device=x.device)

        grad[..., indices_odd] = self.mu * torch.cos(x[..., indices_odd])  # Derivative of M*sin(x_{2i-1})
        grad[..., indices_even] = -torch.sin(x[..., indices_even])   # Derivative of cos(x_{2i})
        grad += ((self.L - self.mu) / n) * (self.ones_matrix @ x.T ).T  # Contribution from quadratic term
        
        return grad



class PycutestTarget(TargetFunction):
    
    def __init__(self, pycutest_problem, dtype = torch.float64, device = 'cpu', seed = 121314, x0 = None) -> None:
        self.fun = pycutest_problem# pycutest.import_problem(problem_name)        
        super().__init__("CUTest", d = self.fun.n, dtype = dtype, device= device, seed=seed)
        if x0 is None:
            self.x0 = torch.from_numpy(self.fun.x0).to(device=device, dtype=dtype).reshape(1, -1)
        else:
            self.x0 = torch.from_numpy(x0).to(device=device, dtype=dtype).reshape(1, -1)
            
            
    def __call__(self, x):
#        print(x.cpu().numpy())
        res = np.apply_along_axis(self.fun.obj, axis=1, arr=x.cpu().numpy())

        fx = torch.from_numpy(res).to(dtype=self.dtype, device=self.device)
        return self.add_noise(fx)

    def grad(self, x):
        return torch.from_numpy(self.fun.grad(x.cpu().numpy().flatten())).to(device=self.device, dtype=self.dtype)
#        _, g = self.fun.obj(x.cpu().numpy().flatten(), gradient = True)
#        return torch.from_numpy(g).to(device=self.device,dtype=self.dtype)
    
    def close(self):
        if hasattr(self.fun, 'close'):
            self.fun.close()