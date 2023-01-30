from torch import Tensor, zeros_like, preserve_format, zeros
from typing import Callable, Any

from eszd.optimizer import Optimizer
from eszd.directions import DirectionMatrix

class ESZD(Optimizer):
    
    def __init__(self, 
                 target : Callable[[Tensor, Any], float],
                 x : Tensor, 
                 P : DirectionMatrix,
                 alpha : Callable[[int], float] = lambda _ : 1.0,
                 h : Callable[[int], float] = lambda _ : 1e-7,
                 verbose : bool = False):
        super().__init__(target, x, verbose)
        self.P = P
        self.alpha = alpha
        self.h = h

        
    def step(self):
        Pk = self.P()
        h = self.h(self.k)
        alpha = self.alpha(self.k)
        fx = self.target(self.x)
        g = Pk.matmul((self.target(self.x + h * Pk.T) - fx).div(h)).flatten()
        self.x.add_(g, alpha=-alpha)
        self.f_evals += Pk.shape[1]
        self.k += 1
        return self.x
    
    
    