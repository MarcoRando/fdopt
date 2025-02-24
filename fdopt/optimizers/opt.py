
import torch
import numpy as np

from typing import Callable, Dict, Tuple, Any
from fdopt.direction_generators import DirectionGenerator
   

class Optimizer:
        
    def __init__(self, x, P, gamma, h) -> None:
        self.x = x
        self.P = P
        self.gamma = gamma
        self.h = h
        
    @property
    def d(self):
        return self.P.d

    @property
    def l(self):
        return self.P.l

