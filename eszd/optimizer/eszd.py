from torch import Tensor, finfo, empty
from typing import Callable, Any

from eszd.optimizer import Optimizer
from eszd.directions import DirectionMatrix

class ESZD(Optimizer):
    
    def __init__(self, 
                 target : Callable[[Tensor], Tensor],
                 x : Tensor, 
                 P : DirectionMatrix,
                 alpha : Callable[[int], float] = lambda _ : 1.0,
                 h : Callable[[int], float] = lambda _ : 1e-7,
                 z : Callable[[Any], Any] = lambda : None,
                 verbose : bool = False):
        super().__init__(target, x, P, verbose)
        self.h = h
        self.alpha = alpha
        self.z = z
        self.min_val = finfo(P.dtype).eps

    def approx_grad(self, Pk, h):
        fx = self.target(self.x, self.z)
        d = self.target(self.x + h * Pk, self.z).add_(fx, alpha=-1).div_(h)
        return Pk.T.mv(d)
              
    def step(self):
        Pk = self.P() # should be l x d
        h = max(self.h(self.k), self.min_val)
        alpha = max(self.alpha(self.k), self.min_val)
        g = self.approx_grad(Pk, h)
        self.x.add_(g, alpha=-alpha)
        self.f_evals += self.l + 1
        self.k += 1
        return self.x.clone()
    
    
    