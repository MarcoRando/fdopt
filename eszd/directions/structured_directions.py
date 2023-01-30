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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.I = torch.eye(self.d, self.d, device=self.device, dtype=self.dtype)

    def __call__(self):
        v = torch.randn(self.d, generator=self.generator, dtype=self.dtype, device=self.device) 
        v.div_(v.norm().square())
        inds = torch.randperm(self.d, device=self.device, generator=self.generator)[:self.l]

        return self.I[:, inds].add(v.outer(v[inds]), alpha=-2)
        

    
class RandomCoords(DirectionMatrix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.I = torch.eye(self.d, self.d, device=self.device, dtype=self.dtype)
    
    def __call__(self):
        inds = torch.randperm(self.d, device=self.device, generator=self.generator)[:self.l]
        return self.I[:, inds]

class FastRandomCoords(DirectionMatrix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.I = torch.eye(self.d, self.d, device=self.device, dtype=self.dtype)
        self.idx = torch.randperm(self.d, device=self.device, generator=self.generator) 
        self.t = 0
    
    def __call__(self):
        if self.t >= self.d // self.l:
            self.t = 0
            self.idx = torch.randperm(self.d, device=self.device, generator=self.generator)
        else:
            self.t += 1
        return self.I[:, self.idx[self.t * self.l : self.l * (self.t+1)]]
