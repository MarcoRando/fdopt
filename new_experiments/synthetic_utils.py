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
        self.A = torch.randn((l_dim, d), device=device, dtype=dtype)
        
    def __call__(self, x, z=None):
        return torch.linalg.norm(x @ self.A.T, dim=1, ord=2).square()


class SparseSphere(TargetFunction):        
    def __init__(self, d: int, l_dim : int, seed : int = 1212, dtype=torch.float64, device : str ='cpu') -> None:
        super().__init__("ScalarProd", d)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.inds = range(l_dim)
        
    def __call__(self, x, z=None):
        return x[:, -1].square()#.sum(dim=1)

