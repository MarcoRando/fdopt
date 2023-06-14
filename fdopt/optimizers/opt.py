
import torch
import numpy as np

from typing import Callable, Dict, Tuple, Any
from fdopt.direction_generators import DirectionGenerator

class Optimizer:
        
    def _build_result(self, x : torch.Tensor, fx : float, trace = None):
        result = {'x' : x, 'fx' : fx}
        if trace is not None:
            result['trace'] = trace
        return result
    
    def optimize(self, f : Callable, x_0 : torch.Tensor, T : int = 100) -> Dict[str, Any]:
        raise NotImplementedError("This method is implemented in subclasses!")
