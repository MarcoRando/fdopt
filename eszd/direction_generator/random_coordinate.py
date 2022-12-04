import numpy as np
from eszd.direction_generator import DirectionMatrix

class RandomCoordinate(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inds = torch.ones(self.d)
    
    
    def __call__(self):
        P = torch.zeros((self.d, self.l), device=self.device, dtype=self.dtype)
        inds = self.inds.multinomial(self.l, replacement=False, generator=self.generator)
        P[inds, range(self.l)] = ((torch.rand(self.l, device=device, generator=self.generator) < 0.5)*2 - 1).to(dtype=self.dtype) 
        return self.nrm_const * P