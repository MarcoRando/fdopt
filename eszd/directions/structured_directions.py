import os
import torch
from eszd.directions.directions import DirectionMatrix

from torch.fft import fft

from torch_dct import dct

from math import pi, sqrt, cos, sin, log2

import numpy as np

class HouseholderQRDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        return  torch.linalg.qr(A, mode='reduced')[0]

class HouseholderReflection(DirectionMatrix):

    def __init__(self, dct = False, **kwargs):
        super().__init__(**kwargs)
        self.I = torch.eye(self.d, self.l, device=self.device, dtype=self.dtype)
        self.mult = torch.ones(self.d, device='cuda')
        self.m_gen = torch.Generator(device='cuda')
        self.m_gen.manual_seed(121212)
        self.lrange = range(self.l)

        self.dct = dct

    def __call__(self):
        v = torch.randn(self.d, generator=self.generator, dtype=self.dtype, device=self.device) 
        v.div_(v.norm().square())
        inds = self.mult.multinomial(self.l, replacement=False, generator=self.m_gen)#.to(self.device)
        P = torch.zeros(self.d, self.l, device=self.device, dtype=self.dtype)
#        inds = torch.randperm(self.d, device=self.device, generator=self.generator)[:self.l]
#        P = self.I.add(v.outer(v[:self.l]), alpha=-2)
        P[inds, self.lrange] = 1.0
        if self.dct:
            P = dct(P.T, norm='ortho').T
        return P#.mul_(self.nrm_const)
        

    
class RandomCoords(DirectionMatrix):
    def __init__(self, dct=False, **kwargs):
        super().__init__(**kwargs)
#        self.I = torch.zeros(self.d, self.l, device=self.device, dtype=self.dtype)
        self.mult = torch.ones(self.d, device='cuda')
        self.m_gen = torch.Generator(device='cuda')
        self.m_gen.manual_seed(121212)
        self.lrange = range(self.l)
        self.dct = dct
        
    def _compute_transform(self, P):
        return dct(P, norm='ortho').T
    
    def __call__(self):
#        inds = torch.randperm(self.d, device=self.device, generator=self.generator)[:self.l]
        inds = self.mult.multinomial(self.l, replacement=False, generator=self.m_gen).to(self.device)
        P = torch.zeros(self.d, self.l, device=self.device, dtype=self.dtype)
 #       self.I[inds, self.lrange] = 1.0
        P[inds, self.lrange] = 1.0
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
#        print(self.d, self.l * self.t)
#        print(self.t, self.l, self.d // self.l, self.idx[:10], self.idx[self.t * self.l : (self.l * (self.t+1))])
        P = self.I[:, self.idx[self.t * self.l : self.l * (self.t+1)]]
        if self.dct:
            P = self._compute_transform(P.T)
            P.div_(P.norm(dim=0))
        return P#.mul_(self.nrm_const)
