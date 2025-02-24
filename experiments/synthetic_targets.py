import torch
import numpy as np 
import pycutest
from math import sqrt

class TargetFunction:
    
    def __init__(self, d, dtype = torch.float64, device = 'cpu', sigma_noise = 0.0, seed = 121314) -> None:
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
        
class LeastSquares(TargetFunction):
    
    def __init__(self, mu, L, set_l = True, *args, **kwargs) -> None:
        assert mu < L
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.L = L
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
        
    def __call__(self, x):
        fx = torch.linalg.norm(self.A @ x.T - self.y, dim=0).square().mul(0.5)
        return self.add_noise(fx)

    def grad(self, x):
        return self.A.T @ (self.A @ x.T - self.y)
    
class LS1(LeastSquares):
    
    def __init__(self, d, *args, **kwargs):
        super().__init__(mu = 1.0, L = 10.0, d=d, set_l=True, *args, **kwargs)
    
class LS2(LeastSquares):
    
    def __init__(self, d, *args, **kwargs):
        super().__init__(mu = 0.1, L = 100, d=d, set_l=True, *args, **kwargs)

class LS3(LeastSquares):
    
    def __init__(self, d, *args, **kwargs):
        super().__init__(mu = 0.0, L = 100, d=d, set_l=True, *args, **kwargs)



class PycutestTarget(TargetFunction):
    
    def __init__(self, d, problem_name, dtype = torch.float64, device = 'cpu', seed = 121314) -> None:
        self.fun = pycutest.import_problem(problem_name)        
        super().__init__(d = self.fun.n, dtype = dtype, device= device, seed=seed)
        self.x0 = torch.from_numpy(self.fun.x0).to(device=device, dtype=dtype).reshape(1, -1)
        
    def __call__(self, x):
#        res = np.zeros((x.shape[0], ))
#        for i in range(x.shape[0]):
#            res[i] = self.fun.obj(x[i, :].cpu().numpy().flatten(), gradient = False)
        res = np.vectorize(self.fun.obj, x.cpu().numpy())
        fx = torch.from_numpy(res).to(dtype=self.dtype, device=self.device)
        return self.add_noise(fx)

    def grad(self, x):
        _, g = self.fun.obj(x.cpu().numpy().flatten(), gradient = True)
        return torch.from_numpy(g).to(device=self.device,dtype=self.dtype)