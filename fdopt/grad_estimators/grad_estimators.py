import torch

from typing import Union, Callable, Dict

from fdopt.direction_generators.generator import DirectionGenerator
from fdopt.direction_generators.structured_directions import CoordinateDirections

class GradEstimator:
    
    def __init__(self, norm_const : float = 1.0):
        self.nrm_const = norm_const
                    
    def __call__(self, f : Callable[[torch.Tensor], torch.Tensor], x : torch.Tensor, P : torch.Tensor, h : float):
        raise NotImplementedError("Method implemented in subclasses")
                

class ForwardFD(GradEstimator):
        
    def __call__(self, f : Callable[[torch.Tensor], torch.Tensor], x : torch.Tensor,  P : torch.Tensor, h : float):        
        return ((f(x + h * P) - f(x.repeat_interleave(P.shape[0], dim=0)) ).div(h).reshape(P.shape[0], -1) * P).sum(dim=0, keepdims=True).mul(self.nrm_const)        

