import time
import torch
from math import floor, log2, pi, sin, cos, sqrt

from torch_dct import dct

from fdopt.direction_generators.generator import DirectionGenerator

class CoordinateDirections(DirectionGenerator):
    
    def __init__(self, d: int, l: int = 1, dtype: torch.dtype = torch.float64, store_identity : bool = True, device: str = 'cpu', seed: int = 121212) -> None:
        super().__init__(d, l, dtype, device, seed)
        self.store_identity = store_identity
        if store_identity:# or d == l:
            self.I = torch.eye(self.d, dtype=self.dtype, device=self.device)
    
    def __call__(self) -> torch.Tensor:
        if self.store_identity and self.l == self.d:
            return self.I
        elif self.store_identity and self.l < self.d:
            inds = torch.multinomial(torch.ones(self.d, dtype=self.dtype, device=self.device), num_samples=self.l, replacement=False, generator=self.generator) 
            return self.I[inds, :]
        I = torch.zeros((self.l, self.d), dtype = self.dtype, device=self.device)
        inds = torch.multinomial(torch.ones(self.d, dtype=self.dtype, device=self.device), num_samples=self.l, replacement=False, generator=self.generator) 
        I[range(self.l), inds] = 1.0
        return I
    
    def __str__(self):
        return f"Coordinate_{self.d}_{self.l}"

class QRDirections(DirectionGenerator):
    def __call__(self) -> torch.Tensor:
        P = torch.randn(self.d, self.l, generator=self.generator, dtype=self.dtype, device = self.device)
#        Q, R = torch.linalg.qr(P)
        return torch.linalg.qr(P)[0].T
#        D = torch.diagonal(R, offset=0)
#        return Q.matmul((D / D.abs_()).diag().matmul(Q.T)).T

    def __str__(self):
        return f"QR_{self.d}_{self.l}"

class SVDDirections(DirectionGenerator):
    def __call__(self) -> torch.Tensor:
        P = torch.randn(self.d, self.l, generator=self.generator, dtype=self.dtype, device = self.device)
        return torch.linalg.svd(P, full_matrices=False)[0].T

    def __str__(self):
        return f"QR_{self.d}_{self.l}"

    
class Householder(DirectionGenerator):
    
    def __init__(self, num_reflectors = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.I = torch.eye(self.d, device=self.device, dtype=self.dtype)
        self.num_reflectors = num_reflectors
            
    def __call__(self) -> torch.Tensor:
        v =  torch.randn(self.d, generator=self.generator, dtype=self.dtype,  device = self.device)
        v /=torch.linalg.norm(v, ord=2)
        H = self.I - 2 * v.outer(v)
        for i in range(1, self.num_reflectors):
            v =  torch.randn(self.d, generator=self.generator, dtype=self.dtype,  device = self.device)
            v /=torch.linalg.norm(v, ord=2)
            H = (self.I - 2 * v.outer(v)) @ H
            
#        sx = v[:self.l].sign() #if self.take_all_sign  else v[0].sign()
      #  print(H[:, :self.l].T)
        return  H[:, :self.l].T #(self.I_l - 2 * v.outer(v[:self.l])).T

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
        d_0 = B.shape[0]
        to_add = self.d - B.shape[0]
        B = torch.vstack([B, torch.zeros(self.d - B.shape[0], B.shape[1], dtype=self.dtype, device=self.device)])
        B = torch.hstack([B, torch.zeros(B.shape[0], to_add, dtype=self.dtype, device=self.device )])
#        if B.shape[1] < self.l:
#            B = torch.hstack([B, torch.zeros(B.shape[0], self.l - B.shape[1], dtype=self.dtype, device=self.device )])
        B[range(d_0, d_0 + to_add), range(d_eff, d_eff + to_add)] = 1.0
        if self.dct:
            return dct(B[:, :self.l].T, norm='ortho')
#        print(B)
        B = B[:, torch.randperm(B.shape[1], generator=self.generator, device=self.device)[:self.l]]
        return B.T
      
    def __str__(self):
        if self.dct:
            return f"ButterflyDCT_{self.d}_{self.l}"            
        return f"Butterfly_{self.d}_{self.l}"