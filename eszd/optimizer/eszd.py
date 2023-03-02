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
                 sampling_fun : Callable[[Any], Any] = lambda : None,
                 single_point : bool = False,
                 verbose : bool = False):
        super().__init__(target, x, P, verbose)
        self.h = h
        self.alpha = alpha
        self.sampling_fun = sampling_fun
        self.single_point = single_point
        self.min_val = finfo(P.dtype).eps
        self.approx_grad = self._single_point if self.single_point else self._two_points

    def _two_points(self, Pk, h, z = None):
        fx = self.target(self.x, z)
        d = self.target(self.x + h * Pk, z).add_(fx, alpha=-1).div_(h)
        return Pk.T.mv(d)
    
    def _single_point(self, Pk, h, z = None):
        d = self.target(self.x + h * Pk, z).div_(h)
        return Pk.T.mv(d)
      
        
    def step(self):
        Pk = self.P() # should be l x d
        h = max(self.h(self.k), self.min_val)
        alpha = max(self.alpha(self.k), self.min_val)
        z = self.sampling_fun()
        g = self.approx_grad(Pk, h, z)
        self.x.add_(g, alpha=-alpha)
     #   print(g)
        self.f_evals += self.l + int(not self.single_point)
        self.k += 1
        return self.x.clone()
    
    
    