import numpy as np

from typing import Callable

    
class Stepsize:
    
    def __init__(self, gamma_0 : float, law : Callable[[int], float]) -> None:
        self.gamma_0 = gamma_0
        self.law = law
        
    def __call__(self, k : int) -> float:
        return self.gamma_0 * self.law(k)


class ConstantStepsize(Stepsize):
    
    def __init__(self, gamma_0 : float) -> None:
        super().__init__(gamma_0 = gamma_0, law=None)
        
    def __call__(self, k : int) -> float:
        return self.gamma_0

        
        
        
        