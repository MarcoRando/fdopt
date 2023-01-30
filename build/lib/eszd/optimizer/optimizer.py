from torch import Tensor
from typing import Callable

class Optimizer:
        
    def __init__(self, target : Callable, x : Tensor, verbose : bool = False) -> None:
        self.target = target
        self.x = x
        self.verbose = verbose
        self.f_evals = 0
        self.k = 1
                
    def step(self):
        pass
    
    
    def reset(self, target : Callable | None = None):
        if target is not None:
            self.target = target
        self.f_evals = 0
        self.k = 1