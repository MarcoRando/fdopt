from torch import Tensor, zeros_like, preserve_format, zeros
from typing import Callable, Any

from eszd.optimizer import Optimizer
from eszd.directions import DirectionMatrix

class ESZD(Optimizer):
    
    def __init__(self, 
                 target : Callable[[Tensor], float],
                 x : Tensor, 
                 P : DirectionMatrix,
                 alpha : Callable[[int], float] = lambda _ : 1.0,
                 h : Callable[[int], float] = lambda _ : 1e-7,
                 verbose : bool = False):
        super().__init__(target, x, P, verbose)
        self.alpha = alpha
        self.h = h


    def _approx_grad(self, fx, Pk, h):
        return Pk.matmul((self.target(self.x + h * Pk.T) - fx).div(h)).flatten()
        
    def step(self):
        Pk = self.P()
        h = self.h(self.k)
        alpha = self.alpha(self.k)
        fx = self.target(self.x)
        g = self._approx_grad(fx, Pk, h) #Pk.matmul((self.target(self.x + h * Pk.T) - fx).div(h)).flatten()
        self.x.add_(g, alpha=-alpha)
        self.f_evals += (self.l + 1)
        self.k += 1
        return self.x
    
    
    