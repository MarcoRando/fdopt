from torch import Tensor, finfo, empty
from typing import Callable, Any

from eszd.optimizer import Optimizer
from eszd.directions import DirectionMatrix

class ESVRZD(Optimizer):
    
    def __init__(self, 
                 target : Callable[[Tensor], float],
                 x : Tensor, 
                 P : DirectionMatrix,
                 sampling_fun : Callable[[Any], Any],
                 alpha : Callable[[int], float] = lambda _ : 1.0,
                 h : Callable[[int], float] = lambda _ : 1e-7,
                 m : int = 10,
                 verbose : bool = False):
        super().__init__(target, x, P, verbose)
        self.alpha = alpha
        self.sampling_fun = sampling_fun
        self.h = h
        self.m = m
        self.min_val = finfo(P.dtype).eps
        

    def _approx_grad(self, fx, Pk, h, z = None):
        g = empty(Pk.shape[1], dtype=self.P.dtype, device=self.P.device)
        for i in range(Pk.shape[1]):
            g[i] = (self.target(self.x + h * Pk[:, i], z) - fx)
        g.div_(h)
        return Pk.mv(g)

        
    def step(self):
        P_tau = self.P()
        h = max(self.h(self.k), self.min_val)
        alpha = max(self.alpha(self.k), self.min_val)
        f_tau = self.target(self.x)
        G = self._approx_grad(f_tau, P_tau, h, z=None) 
#        print(G)
        x_t = self.x.clone()
        for t in range(self.m):
            P_k = self.P()
            z_k = self.sampling_fun()
            f_k = self.target(self.x, z_k)
            g_tau = self._approx_grad(self.target(self.x, z_k), P_tau, h, z=z_k)
            g_k = self._approx_grad(f_k, P_k, h, z=z_k)
            v_k = G + (g_k - g_tau)
   #         print(g_k - g_tau)
   #         exit()
            x_t.add_(v_k, alpha=-alpha) 
        self.x = x_t            
        self.f_evals += ((2*self.l + 2)*self.m + (self.l + 1))
        self.k += 1
        return self.x.clone()
    
    
    