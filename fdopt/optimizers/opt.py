
import torch
import numpy as np

from typing import Callable, Dict, Tuple, Any
from fdopt.direction_generators import DirectionGenerator
   

class Optimizer:
        
    def _build_result(self, x : torch.Tensor, fx : float, k : int, trace = None, converged = None):
        result = {'x' : x, 'fx' : fx, 'iterations' : k}
        if trace is not None:
            result['trace'] = trace
        if converged is not None:
            result['converged'] = converged
        return result
    
    def optimize(self, f : Callable, x_0 : torch.Tensor, T : int | None = None, convergence_criterion : Callable | None = None) -> Dict[str, Any]:
        raise NotImplementedError("This method is implemented in subclasses!")

    def reset(self):
        pass