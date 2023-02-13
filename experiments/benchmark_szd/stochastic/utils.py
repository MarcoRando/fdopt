import torch
import numpy as np

class SquareFun:
    
    def __init__(self, d = 1, seed=12, dtype=torch.float32, device='cpu') -> None:
        self.d = d
        self.device= device
        self.dtype = dtype
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.A = torch.randn((d, d), generator=self.gen, dtype=self.dtype)
        
    def __call__(self, x, z = None):
        if z is not None:
            return (self.A[z, :] @ x).square()
        return 1/self.d * (self.A @ x).norm().square()


class LinearRegr:
    
    def __init__(self, num_points = 100, d = 1, seed = 12343567, device = 'cpu', dtype=torch.float32):
        self.num_points = num_points
        self.d  = d
        self.device = device
        self.dtype = dtype
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.X = torch.rand((num_points, d), generator=self.gen, dtype=dtype, device=self.device)
        self.w = torch.randn(d, dtype=dtype, device=self.device, generator=self.gen)
        self.y = self.X.mv(self.w)#@ self.w.T
        
    def __call__(self, w, z = None):
        if z is not None:
            return (self.X[z, :].matmul(w) - self.y[z]).square()
        return  (self.X.matmul(w) - self.y).square().mean()

class LinearRegrNumpy(LinearRegr):
    
    def __call__(self, w, z = None):
        w_t = torch.from_numpy(w).to(self.device)
        return super().__call__(w_t, z=z).to('cpu').numpy()
    
        

