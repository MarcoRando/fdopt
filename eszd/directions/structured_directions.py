import os
import torch
from eszd.directions.directions import DirectionMatrix
from torch.fft import fft

from torch_dct import dct

from math import pi, sqrt, cos, sin, log2, floor

import numpy as np

class HouseholderQRDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        return  torch.linalg.qr(A, mode='reduced')[0].T


@torch.jit.script
def multiply_reflections(R1 : torch.Tensor, R2 : torch.Tensor) -> torch.Tensor:
    return R1.matmul(R2)

@torch.jit.script
def build_reflection(I : torch.Tensor, v : torch.Tensor, l : int) -> torch.Tensor:
    return (I - 2 * v.outer(v[:l]))

class HouseholderReflection(DirectionMatrix):

    def __init__(self, dct = False, permutation=False, num_reflection=1, **kwargs):
        super().__init__(**kwargs)
        self.mult = torch.ones(self.d, device=self.device)
        self.m_gen = torch.Generator(device=self.device)
        self.m_gen.manual_seed(121212)
        self.num_reflection = num_reflection
        self.permutation=permutation
        self.lrange = range(self.l)

        self.dct = dct

    def _generate_reflection(self, v):
        inds = self.mult.multinomial(self.l, replacement=False, generator=self.m_gen)#.to(self.device)
        P = torch.zeros(self.d, self.l, device=self.device, dtype=self.dtype)
        P[inds, self.lrange] = 1.0
        return build_reflection(P, v, self.l)
        

    def __call__(self):
        v = torch.rand(self.d, self.num_reflection, generator=self.generator, dtype=self.dtype, device=self.device) 
        v.div_(v.norm(dim=0))
        R = self._generate_reflection(v[:, 0])
       # print(R.shape)
        return  R.T
       

    
class RandomCoords(DirectionMatrix):
    def __init__(self, dct=False, **kwargs):
        super().__init__(**kwargs)
#        self.I = torch.zeros(self.d, self.l, device=self.device, dtype=self.dtype)
        self.mult = torch.ones(self.d, device=self.device)
        self.m_gen = torch.Generator(device=self.device)
        self.m_gen.manual_seed(121212)
        self.lrange = range(self.l)
        self.dct = dct
        
    def _compute_transform(self, P):
        return dct(P, norm='ortho').T
    
    def __call__(self):
        inds = self.mult.multinomial(self.l, replacement=False, generator=self.m_gen).to(self.device)
        P = torch.zeros(self.l, self.d, device=self.device, dtype=self.dtype)
        P[self.lrange, inds] = 1.0
        if self.dct:
            P = self._compute_transform(P.T)
        return P#.mul_(self.nrm_const)

class FastRandomCoords(DirectionMatrix):
    def __init__(self, dct = False, **kwargs):
        super().__init__(**kwargs)
        self.I = torch.eye(self.d, self.d, device = self.device, dtype = self.dtype)
        self.idx = torch.randperm(self.d, device=self.device, generator=self.generator) 
        self.t = 0
        self.dct = dct
        
    def _compute_transform(self, P):
        return dct(P, norm='ortho').T
    
    def __call__(self):
        if (self.t + 1) >= (self.d // self.l):
            self.t = 0
            self.idx = torch.randperm(self.d, device=self.device, generator=self.generator)
        else:
            self.t += 1
        P = self.I[self.idx[self.t * self.l : self.l * (self.t+1)], :]
        if self.dct:
            P = self._compute_transform(P.T)
            P.div_(P.norm(dim=0))
        return P#.mul_(self.nrm_const)
    
class HBDirections(DirectionMatrix):
    
    def __init__(self, dct=True, permute=False, **kwargs):
        super().__init__(**kwargs)
        self.k = floor(log2(self.d))
        self.ek = (1 << self.k)
        self.dct = dct
        self.permute = permute
        
    def __call__(self):
        thetas = 2 * pi * torch.rand(self.k, generator=self.generator, device=self.device)
        A = torch.eye(1, dtype=self.dtype, device=self.device)
        #k = floor(log2(self.d))
        num_additional = self.d - self.ek
        for i in range(self.k):
            theta = thetas[i]
            A_cut = A[:, :self.l]
            c, s = cos(theta) * A_cut, sin(theta) * A_cut
            A = torch.vstack((torch.hstack([c, -s]), torch.hstack([s, c]))) #torch.hstack([c, -s, s, c]).reshape(2*c.shape[0], -1)

        A = A[:, :self.l]
        if num_additional == 0:
            return self.nrm_const * A
        A = torch.vstack((A, torch.zeros(num_additional, A.shape[1], dtype=self.dtype, device=self.device)))
        A = torch.hstack((A, torch.zeros(A.shape[0], num_additional, dtype=self.dtype, device=self.device)))
        A[self.ek:, self.l:].fill_diagonal_(1.0)
        if self.dct:
            A = dct(A.T, norm='ortho').T
        if self.permute:
            inds = torch.randperm(A.shape[1], generator=self.generator)[:self.l]
            return A[:, inds]
            

        return A[:, :self.l]
        idx =range(-1, -num_additional -1, -1)
        A[idx, idx] = 2 * (torch.rand(num_additional, dtype=self.dtype, generator=self.generator, device=self.device) > 0.5).to(self.dtype) -1.0
        #for i in range(1, num_additional + 1):
        #    A[-i, -i] = 2 * (torch.rand(1, dtype=self.dtype, generator=self.generator) > 0.5).to(self.dtype) -1.0
        inds = torch.randperm(A.shape[1], device=self.device, generator=self.generator)
        A = A[:, inds]
        return self.nrm_const * A[:, :self.l]

class HaarButterflyDirections(DirectionMatrix):
    def __init__(self, dct = True, permute=False, **kwargs):
        super().__init__(**kwargs)
#        self.I = torch.eye(self.d, self.d, device = self.device, dtype = self.dtype)
#        self.idx = torch.randperm(self.d, device=self.device, generator=self.generator) 
        self.t = 0
        self.k = floor(log2(self.d))
        self.ek = (1 << self.k )
        self.n_additional = self.d - self.ek
        self.dct = dct
        self.permute = permute
        self.btt = Butterfly(self.ek, self.l, init='ortho', dtype=self.dtype)
        self.I = torch.eye(self.ek, device=self.device, dtype=self.dtype)

#        if self.n_additional > 0:
#            self.add_row = torch.zeros((self.n_additional, self.l), dtype=self.dtype, device=self.device)
#            self.add_col = torch.zeros((self.d, self.n_additional), dtype=self.dtype, device=self.device)
                
    def __call__(self):
#        idx = torch.randperm(self.d, device=self.device, generator=self.generator)         k = 
        P = self.btt(self.I)
        if self.n_additional > 0:
            P = torch.vstack((P, torch.zeros((self.n_additional, self.l), dtype=self.dtype, device=self.device)))
            P = torch.hstack((P, torch.zeros((self.d, self.n_additional), dtype=self.dtype, device=self.device)))
            P[self.ek:, self.l:].fill_diagonal_(1.0)
        if self.dct:
            P = dct(P.T, norm='ortho').T
        self.btt.reset_parameters()
        if self.permute:
            inds = torch.randperm(P.shape[1], generator=self.generator, device=self.device)[:self.l]
            return P[:, inds]
        return P[:, :self.l]