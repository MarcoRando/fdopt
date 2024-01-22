
import torch
import numpy as np

from typing import Callable, Dict, Tuple, Any
from fdopt.direction_generators import DirectionGenerator
   

class Optimizer:
        
    def __init__(self, target, x, P, gamma, theta, rho, armijo_constant, min_gamma, max_gamma, h, seed = 121314) -> None:
        self.target = target
        self.x = x
        self.P = P
        self.theta = theta
        self.rho = rho
        self.gamma = gamma
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.armijo_constant = armijo_constant
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

    def reset(self):
        self.t += 1