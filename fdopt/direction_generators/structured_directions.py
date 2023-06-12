#import numpy as np
import torch
from fdopt.direction_generators.generator import DirectionGenerator

class CoordinateDirections(DirectionGenerator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mult = torch.ones(self.d)

    def __call__(self) -> torch.Tensor:
        I = torch.zeros((self.l, self.d), dtype=self.dtype, device = self.device)
        inds = self.mult.multinomial(num_samples=self.l, replacement=False, generator=self.generator)
        I[range(self.l), inds] = 1.0
        return I

class QRDirections(DirectionGenerator):
    def __call__(self) -> torch.Tensor:
        P = torch.randn(size=(self.d, self.l), generator=self.generator,  dtype=self.dtype, device = self.device)
        return torch.linalg.qr(P)[0].T