import torch


class TargetFunction:
    
    def __init__(self, d, dtype = torch.float64, device = 'cpu', seed = 121314) -> None:
        self.d = d
        self.dtype = dtype
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
    
    def __call__(self, x):
        pass
    
class SumOfSquares(TargetFunction):
    
    def __call__(self, x):
        return x.square().sum(dim=1)
    
    def grad(self, x):
        return x.mul(2.0)
    
class LeastSquares(TargetFunction):
    
    def __init__(self, mu, L, *args, **kwargs) -> None:
        assert mu < L
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.L = L
        Q = torch.randn(self.d, self.d, generator=self.generator, dtype=self.dtype, device=self.device)
        U, S, V = torch.svd(Q)
        S = torch.diag(torch.linspace(L, mu, self.d, dtype=self.dtype, device=self.device))
        self.A = U @ S @ V       
        self.x_star = torch.randn(1, self.d, generator=self.generator, device=self.device, dtype=self.dtype) 
        self.y = self.A @ self.x_star.T
        
    def __call__(self, x):
        x = x.T
        return torch.linalg.norm(self.A @ x - self.y, dim=0).square().mul(0.5)

    def grad(self, x):
        x = x.T
        return self.A.T @ (self.A @ x - self.y)
    
class Rosenbrock(TargetFunction):

    def __call__(self, x : torch.Tensor):
        return torch.sum((1.0 - x[:, :-1])**2 + 100.0 * (x[:, 1:] - x[:, :-1]**2)**2, dim=1)

    def grad(self, x : torch.Tensor):
        a, b = 1.0, 100.0
        df_dx = torch.zeros_like(x)
        df_dx[:, :-1] += -2 * (a - x[:, :-1]) - 4 * b * x[:, :-1] * (x[:, 1:] - x[:, :-1]**2)
        df_dx[:, 1:] += 2 * b * (x[:, 1:] - x[:, :-1]**2)
        return df_dx
    
class Rastrigin(TargetFunction):
    
    def __call__(self, x):
        return 10 * x.shape[1] + torch.sum(x**2 - 10 * torch.cos(2 * torch.pi * x), dim=1)

    def grad(self, x):
        return 2 * (x + 10 * torch.pi * torch.sin(2 * torch.pi * x))
    
class StyblinskiTang(TargetFunction):
    
    def __call__(self, x):
        return  0.5 * torch.sum( (x**4) - 16 * (x**2) + 5 * x, dim=1)

    def grad(self, x):
        return 2 * (x**3) - 16 *x + (5/2)

class Trid(TargetFunction):
    
    def __call__(self, x):
        d = x.shape[1]
        term1 = torch.sum((x - 1)**2, dim=1)
        term2 = torch.sum(x[:, :-1] * x[:, 1:], dim=1)
        return term1 - term2

    def grad(self, x):
        d = x.shape[1]
        df_dx = torch.zeros_like(x)
        df_dx[:, 0] = 2 * (x[:, 0] - 1) - x[:, 1]
        df_dx[:, -1] = 2 * (x[:, -1] - 1) - x[:, -2]        
        for i in range(1, d-1):
            df_dx[:, i] = 2 * (x[:, i] - 1) - x[:, i-1] - x[:, i+1]
        return df_dx

