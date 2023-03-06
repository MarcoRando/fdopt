from torch import Tensor, finfo, empty, zeros
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
#        print("\t[ESZD] evals:")
#        g = zeros((1, self.d), dtype=self.P.dtype, device=self.P.device)
#        for i in range(Pk.shape[0]):
#            g += (self.target(self.x + h * Pk[i, :]) - fx).div(h).item() * Pk[i,:]
#            print("i: {}, x: {}\tPK: {}".format(i, self.x, Pk[i, :]))
#            dd = self.target(self.x + h * Pk[i, :]) - fx
#            print("\t\t", dd)
#            print("\t\t",dd/h)
#            print("-"*33)
        return Pk.T.mv(d)
              
    def step(self):
        Pk = self.P() # should be l x d
        h = max(self.h(self.k), self.min_val)
        alpha = max(self.alpha(self.k), self.min_val)
        g = self.approx_grad(Pk, h)
#        print("[ESZD] x: {}\tg: {}\talpha: {}".format(self.x, g, alpha))
        self.x.add_(g, alpha=-alpha)
        self.f_evals += self.l + 1
        self.k += 1
        return self.x.clone()
    
    
    