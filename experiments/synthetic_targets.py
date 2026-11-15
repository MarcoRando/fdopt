import torch
import numpy as np 
from math import sqrt


# from sklearn.datasets import load_svmlight_file


# def load_libsvm_data(datapath, device = 'cpu', dtype = torch.float32):
#     X, y = load_svmlight_file(datapath)
#     X = torch.from_numpy(X.toarray()).to(dtype=dtype,device=device)
#     y = torch.tensor(y, device=device, dtype=dtype)
#     return X, y


class TargetFunction:
    
    def __init__(self, name, d, dtype = torch.float64, device = 'cpu',  seed = 121314) -> None:
        self.name = name
        self.d = d
        self.dtype = dtype
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        
    def __call__(self, x):
        pass
        
    def grad(self, x):
        pass
        
    def close(self):
        pass
        
    def __str__(self):
        return f"{self.name}_{self.d}"

class LeastSquares(TargetFunction):
    
    def __init__(self, d, mu, L, set_l = True, dtype = torch.float64, device = 'cpu',   seed = 121314) -> None:
        assert mu < L
        super().__init__(name="LeastSquares", d = d, dtype = dtype, device= device,  seed=seed)
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
        return  fx

    def grad(self, x):
        return self.A.T @ (self.A @ x.T - self.y)

    def __str__(self):
        return f"{self.name}_{self.L}_{self.mu}_{self.d}"
    

class RegularizedLogistic(TargetFunction):
    
    def __init__(self, d, n, lam = 1e-5, dtype=torch.float64, device='cpu',   seed=121314):
        super().__init__("LogisticReg", d, dtype, device , seed)
        self.X = torch.randn((n, d), dtype=dtype, device=device, generator=self.generator) 
        self.w_star = torch.randn((d,), generator=self.generator, dtype=dtype, device=device)
        self.y = (self.X @ self.w_star).sign()
        self.bounds = torch.tensor([[-1, 1] for _ in range(d)], device=device, dtype=dtype)
                
        self.min_f = 0.0
        self.x0 = torch.full((1, self.d), -1.0, dtype=self.dtype, device=self.device)
        
        self.lam = lam
    
    def __call__(self, w):
        expn = (1 + (-torch.matmul(w, self.X.T) * self.y).exp_()).log_().mean(dim=1)
        return expn + self.lam * w.norm(p=2, dim=1).square()
    
    def grad(self, w):
        num = self.X * -self.y.view(-1, 1) # n x d
        den = (1 + (torch.matmul(w, self.X.T) * self.y).exp_()).view(w.shape[0], self.X.shape[0], 1) # l x n
        t1 =  (num/den).mean(1)
        t2 = 2 * self.lam * w
        
        return t1 + t2
    def __str__(self):
        return f"{self.name}_{self.X.shape[0]}_{self.d}"
    


class PycutestTarget(TargetFunction):
    
    def __init__(self, pycutest_problem, dtype = torch.float64, device = 'cpu', seed = 121314, x0 = None) -> None:
        self.fun = pycutest_problem# pycutest.import_problem(problem_name)        
        super().__init__("CUTest", d = self.fun.n, dtype = dtype, device= device, seed=seed)
        if x0 is None:
            self.x0 = torch.from_numpy(self.fun.x0).to(device=device, dtype=dtype).reshape(1, -1)
        else:
            self.x0 = torch.from_numpy(x0).to(device=device, dtype=dtype).reshape(1, -1)
            
            
    def __call__(self, x):
        res = np.apply_along_axis(self.fun.obj, axis=1, arr=x.cpu().numpy())

        fx = torch.from_numpy(res).to(dtype=self.dtype, device=self.device)
        return  fx

    def grad(self, x):
        return torch.from_numpy(self.fun.grad(x.cpu().numpy().flatten())).to(device=self.device, dtype=self.dtype)

    def close(self):
        if hasattr(self.fun, 'close'):
            self.fun.close()



class RosenbrockFunction(TargetFunction):
    def __init__(self, d, dtype=torch.float64, device='cpu', seed=121314):
        super().__init__("Rosenbrock", d, dtype, device,  seed)
        self.bounds = torch.tensor([[-5.0, 10.0] for _ in range(d)], device=device, dtype=dtype)
        self.min_f = 0.0
        self.x0 = torch.full((1, self.d), 0.5, dtype=self.dtype, device=self.device)
            

    def __call__(self, x):
        x = x if x.ndim > 1 else x.unsqueeze(0) 
        xi = x[:, :-1]
        xi1 = x[:, 1:]
        return (100 * (xi1 - xi**2)**2 + (1 - xi)**2).sum(dim=1).squeeze()

    def grad(self, x):
        x = x if x.ndim > 1 else x.unsqueeze(0) 
        batch_size, d = x.shape
        grad = torch.zeros_like(x)

        xi = x[:, :-1]
        xi1 = x[:, 1:]
        grad[:, :-1] += -400 * xi * (xi1 - xi**2) - 2 * (1 - xi)
        grad[:, 1:] += 200 * (xi1 - xi**2)

        return grad.squeeze()


class QuingFunction(TargetFunction):
    def __init__(self, d, dtype=torch.float64, device='cpu',   seed=121314):
        super().__init__("Quing", d, dtype, device , seed)
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





class GriewankFunction(TargetFunction):
    def __init__(self, d, dtype=torch.float64, device='cpu',   seed=121314):
        super().__init__("Griewank", d, dtype, device , seed)
        self.bounds = torch.tensor([[-600.0, 600.0] for _ in range(d)], dtype=self.dtype, device=self.device)
        self.min_f = 0.0
        self.x0 = torch.full((1, self.d), 1.0, dtype=self.dtype, device=self.device)

    def __call__(self, x):
        d = x.shape[-1]
        indices = torch.arange(1, d + 1, dtype=x.dtype, device=x.device)
        sum_sq = torch.sum(x ** 2, dim=-1) / 4000
        prod_cos = torch.prod(torch.cos(x / torch.sqrt(indices)), dim=-1)
        return 1 + sum_sq - prod_cos

    def grad(self, x):
        d = x.shape[-1]
        indices = torch.arange(1, d + 1, dtype=x.dtype, device=x.device)
        sqrt_i = torch.sqrt(indices)

        x_div_sqrt = x / sqrt_i
        cos_term = torch.cos(x_div_sqrt)
        sin_term = torch.sin(x_div_sqrt)

        prod_cos = torch.prod(cos_term, dim=-1, keepdim=True)  # shape (..., 1)

        grad_sum = x / 2000  
        grad_prod = prod_cos * sin_term / cos_term / sqrt_i  # shape (..., d)
        
        return grad_sum + grad_prod  # full gradient



class TridFunction(TargetFunction):

    def __init__(self, d, dtype=torch.float64, device='cpu',  seed=121314):
        super().__init__("Trid", d, dtype, device,   seed)
        self.bounds = torch.tensor([[-d**2, d**2] for _ in range(d)], device=device, dtype=dtype)
        self.min_f = - (d * (d + 4) * (d - 1)) / 6
        x0 = torch.zeros((1, d), dtype=dtype, device=device)


        self.x0 = x0 #torch.ones((1, self.d), dtype=self.dtype, device=self.device)
    
    def __call__(self, x):
        term1 = (x - 1).square().sum(dim=1, keepdim=True)  
        term2 = (x[:, 1:] * x[:, :-1]).sum(dim=1, keepdim=True)  
        
        return term1 - term2  # Compute function

    def grad(self, x):
        n, d = x.shape
        grad = 2 * (x - 1)  
        
        grad[:, 0] -= x[:, 1]

        grad[:, -1] -= x[:, -2]

        if d > 2:
            grad[:, 1:-1] -= x[:, :-2] + x[:, 2:]

        return grad  # (n, d)