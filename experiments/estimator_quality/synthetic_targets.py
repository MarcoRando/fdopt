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