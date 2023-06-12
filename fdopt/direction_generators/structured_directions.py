import numpy as np
#import torch
from fdopt.direction_generators.generator import DirectionGenerator

class CoordinateDirections(DirectionGenerator):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self) -> np.ndarray:
        I = np.zeros((self.l, self.d), dtype=self.dtype)
        inds = self.rnd_state.choice(self.d, self.l, replace=False) 
        I[range(self.l), inds] = 1.0
        return I

class QRDirections(DirectionGenerator):
    def __call__(self) -> np.ndarray:
        P = self.rnd_state.randn(self.d, self.l)
        return np.linalg.qr(P)[0].T