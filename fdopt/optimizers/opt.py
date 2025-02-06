
import torch
import numpy as np

from typing import Callable, Dict, Tuple, Any
from fdopt.direction_generators import DirectionGenerator
   

class Optimizer:
        
    def __init__(self, x, P, gamma, h, seed = 121314) -> None:
        self.x = x
        self.P = P
        self.gamma = gamma
        self.h = h
        self.t = 1
        self.generator = torch.Generator(P.device)
        self.generator.manual_seed(seed)
        
    @property
    def d(self):
        return self.P.d

    @property
    def l(self):
        return self.P.l

    def ask(self):
        raise NotImplementedError
    
    def tell(self, X, y):
        raise NotImplementedError


    def reset(self):
        self.t += 1