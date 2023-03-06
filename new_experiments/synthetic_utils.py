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
        return x[:, self.inds].square().sum(dim=1)

class Rosenbrock(TargetFunction):
    def __init__(self, d: int) -> None:
        super().__init__("Rosenbrock", d)
    
    def __call__(self, x, z=None):
#        print("SHAPE: ", x.shape)
#        print("x: {}".format(x))
        ris = torch.zeros((x.shape[0]), dtype = x.dtype, device=x.device)
        for i in range(self.d - 1):
            t1 = 100 * ((x[:, i + 1] - (x[:, i]**2)) **2)
            t2 = (x[:, i] - 1)**2
            ris += (t1 + t2)
        return ris
        