from torch import Tensor
from typing import Callable

from eszd.directions import DirectionMatrix

class Optimizer:
        
    def __init__(self, target : Callable, x : Tensor, P : DirectionMatrix, verbose : bool = False) -> None:
        self.target = target
        self.x = x
        self.P = P
        self.verbose = verbose
        self.f_evals = 0
        self.k = 1

    @property
    def d(self):
        return self.P.d
                
    @property
    def l(self):
        return self.P.l
    
    
           
    def step(self):
        pass
    
    
    def reset(self, target : Callable | None = None):
        if target is not None:
            self.target = target
        self.f_evals = 0
        self.k = 1