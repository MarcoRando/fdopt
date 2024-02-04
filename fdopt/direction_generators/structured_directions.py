import time
import torch
from math import floor, log2, pi, sin, cos, sqrt

#from itertools import combinations
#from torch._C import dtype, float64
from torch_dct import dct

from fdopt.direction_generators.generator import DirectionGenerator
from fdopt.direction_generators.unstructured_directions import SphericalDirections, GaussianDirections

class CoordinateDirections(DirectionGenerator):
    
    def __init__(self, d: int, l: int = 1, dtype: torch.dtype = torch.float64, store_identity : bool = False, device: str = 'cpu', seed: int = 121212) -> None:
        super().__init__(d, l, dtype, device, seed)
        self.store_identity = store_identity
        if store_identity:# or d == l:
            self.I = torch.eye(self.d, dtype=self.dtype, device=self.device)
    
    def __call__(self) -> torch.Tensor:
        if self.store_identity:# or self.l == self.d:
            return self.I
        elif self.store_identity and self.l < self.d:
            inds = torch.multinomial(torch.ones(self.d, dtype=self.dtype), num_samples=self.l, replacement=False, generator=self.generator) 
            return self.I[inds, :]
        I = torch.zeros((self.l, self.d), dtype = self.dtype, device=self.device)
        inds = torch.multinomial(torch.ones(self.d, dtype=self.dtype), num_samples=self.l, replacement=False, generator=self.generator) 
        I[range(self.l), inds] = 1.0
        return I
    
    def __str__(self):
        return f"Coordinate_{self.d}_{self.l}"

class QRDirections(DirectionGenerator):
    def __call__(self) -> torch.Tensor:
        P = torch.randn(self.d, self.l, generator=self.generator, dtype=self.dtype, device = self.device)
        return torch.linalg.qr(P)[0].T

    def __str__(self):
        return f"QR_{self.d}_{self.l}"
    
class Householder(DirectionGenerator):
    
    def __init__(self, num_reflection : int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.I_l = torch.eye(self.l, self.d, device=self.device, dtype=self.dtype)
        if num_reflection > 1:
            self.I = torch.eye(self.d, self.d, device=self.device, dtype=self.dtype)
        self.num_reflection = num_reflection
            
    def __call__(self) -> torch.Tensor:
        v =  torch.randn(self.d, generator=self.generator, device = self.device, dtype=self.dtype)
        v /= torch.linalg.norm(v, ord=2)
        return self.I_l - 2 * torch.outer(v, v[:self.l]).T# - A.T))[:self.l, :]

    def __str__(self):
        return f"Householder_{self.d}_{self.l}"
        
class ButterflyMatrix(DirectionGenerator):
    
    def __init__(self, dct : bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dct = dct
            
    def __call__(self) -> torch.Tensor:
        B = torch.ones(1, 1, dtype=self.dtype, device=self.device)
        num_iters = floor(log2(self.d))
        thetas = 2 * pi * torch.rand(num_iters, generator = self.generator, device=self.device, dtype=self.dtype)
        for i in range(num_iters):
            c, s = torch.cos(thetas[i]) * B[:, :self.l], torch.sin(thetas[i]) * B[:, :self.l]
            B1 = torch.vstack([ c , s])
            B2 = torch.vstack([-s, c])
            B = torch.hstack((B1, B2))
        B = B[:, :self.l]
        if self.d == B.shape[0]:
            return B[:, :self.l].T
        d_eff = B.shape[1]  
        B = torch.vstack([B, torch.zeros(self.d - B.shape[0], B.shape[1])])
        if B.shape[1] < self.l:
            B = torch.hstack([B, torch.zeros(B.shape[0], self.l - B.shape[1] )])
            B[range(d_eff, self.l), range(d_eff, self.l)] = 1.0

        return dct(B.T, norm='ortho') if self.dct else B.T
      
    def __str__(self):
        if self.dct:
            return f"ButterflyDCT_{self.d}_{self.l}"            
        return f"Butterfly_{self.d}_{self.l}"