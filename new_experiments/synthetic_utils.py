import torch

class TargetFunction:
    
    def __init__(self, name, d : int) -> None:
        self.name = f"{name}_{d}"
        self.d = d
        
    def __call__(self, x, z = None):
        pass
        


class Sphere(TargetFunction):
    
    def __init__(self, d: int) -> None:
        super().__init__("Sphere", d)
        
    def __call__(self, x, z=None):
        return x.square().sum(dim=1)
        

class SparseMatrix(TargetFunction):        
    def __init__(self, d: int, l_dim : int, seed : int = 1212, dtype=torch.float64, device : str ='cpu') -> None:
        super().__init__("SparseMatrix", d)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.A = torch.randn((l_dim, d), device=device, dtype=dtype, generator=self.gen)
        
    def __call__(self, x, z=None):
        return torch.linalg.norm(x @ self.A.T, dim=1, ord=2).square()


class SparseSphere(TargetFunction):        
    def __init__(self, d: int, l_dim : int, seed : int = 1212, dtype=torch.float64, device : str ='cpu') -> None:
        super().__init__("ScalarProd", d)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.inds = torch.ones(d, dtype=dtype, device = device).multinomial(l_dim, generator=self.gen, replacement=False)
        
    def __call__(self, x, z=None):
        if len(self.inds) == 1:
            return x[:, self.inds].square()
        return x[:, self.inds].square().sum(dim=1)

class Rosenbrock(TargetFunction):
    def __init__(self, d: int, noise_std : float = 0.0, device = 'cpu', dtype = torch.float32,seed = 1212) -> None:
        super().__init__("Rosenbrock", d)
        self.noise_std = noise_std
        self.gen = torch.Generator(device = device)
        self.dtype = dtype
        self.device = device
        self.gen.manual_seed(seed)
    
    def __call__(self, x, z=None):
        ris = torch.zeros((x.shape[0]), dtype = x.dtype, device=x.device)
        for i in range(self.d - 1):
            t1 = 100 * ((x[:, i + 1] - (x[:, i]**2)) **2)
            t2 = (x[:, i] - 1)**2
            ris += (t1 + t2)
        if self.noise_std > 0.0:
            eps = self.noise_std * torch.randn(x.shape[0], dtype=self.dtype, device=self.device)
            return ris + eps
        return ris
        
    def get_exact(self, x):
        ris = torch.zeros((x.shape[0]), dtype = x.dtype, device=x.device)
        for i in range(self.d - 1):
            t1 = 100 * ((x[:, i + 1] - (x[:, i]**2)) **2)
            t2 = (x[:, i] - 1)**2
            ris += (t1 + t2)
        return ris
            
class LinearRegr(TargetFunction):
    
    def __init__(self, d : int, device ='cpu', n_points = 1000, dtype=torch.float32, seed=121212):
        self.d = d
        self.device = device
        self.gen = torch.Generator(device)
        self.dtype = dtype
        self.gen.manual_seed(seed)
        self.w = torch.ones((1, self.d), dtype=dtype, device = device)
        self.X = torch.randn((n_points, self.d), dtype=dtype, generator=self.gen, device=device)
        self.y = self.X @ self.w
        self.n_points = n_points
        
    def __call__(self, x, z = None):
        idx = torch.randint(low=0, high=self.n_points, size=(1,), generator=self.gen)[0]
        return torch.linalg.norm(self.X[idx, :] @ x.T - self.y[idx], dim=1, ord=2).square().mean()
        
#    def get_exact(self, x):
#        return torch.linalg.norm(self.X @ x - self.y, dim=1, ord=2).square()
        
            
            
class DixonPrice(TargetFunction):
    def __init__(self, d : int, noise_std : float = 0.0, device = 'cpu', dtype = torch.float32,seed = 1212):
        super().__init__("DixonPrice", d)
        self.noise_std = noise_std
        self.gen = torch.Generator(device = device)
        self.dtype = dtype
        self.device = device

        self.gen.manual_seed(seed)

    def __call__(self, x, z=None):
        ris = torch.zeros((x.shape[0]), dtype = x.dtype, device=x.device)
        t1 = (x[:, 0] - 1)**2
        for i in range(1, self.d):
            ris += (i + 1) * ((2 * x[:, i].square() - x[:, i - 1]).square())
        ris += t1
        if self.noise_std > 0.0:
            eps = self.noise_std * torch.randn(x.shape[0], dtype=self.dtype, device=self.device)
            return ris + eps
        return ris
