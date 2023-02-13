from torch import Tensor, finfo, empty
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
                 sampling_fun : Callable[[Any], Any] = None,
                 verbose : bool = False):
        super().__init__(target, x, P, verbose)
        self.alpha = alpha
        self.sampling_fun = sampling_fun
        self.h = h
        self.min_val = finfo(P.dtype).eps


    def _approx_grad(self, fx, Pk, h, z = None):
#        if self.k==2:
#            print("[--] h: {}\tfx: {}\tz: {}".format(h, fx, z))
#            print("[--] Pk[:, i] = {}".format(Pk[:, 0].norm()))
#            print("[--] x: {}\tx + hp: {}".format(self.x, self.x + h*Pk.T[0, :]))
#            print("[--] x: {}\tx + hp: {}".format(self.x, self.x + h*Pk.T[1, :]))
#            print("[--] fx: {}\tfx + hp: {}".format(self.target(self.x, z), self.target(self.x + h*Pk[:, 0], z)))
#            print((self.target(self.x + h*Pk.T[0, :], z) -fx)/h)
#            print((self.target(self.x + h*Pk.T[1, :], z) -fx)/h)
#            print(self.x + h * Pk.T, Pk.T)
#          #  print
#            print("[--] fx: {}".format(self.target(self.x + h * Pk.T, z) ))
#            print("[--] fn - fx: {}".format( self.target(self.x + h * Pk.T, z) - fx))
#            print("[--] fn - fx/h: {}".format( (self.target(self.x + h * Pk.T, z) - fx)/h))
#            print("[--] fn - fx/hp: {}".format( Pk.mv(((self.target(self.x + h * Pk.T, z) - fx)/h).flatten())))

           # exit()
        g = empty(Pk.shape[1], dtype=self.P.dtype, device=self.P.device)
        for i in range(Pk.shape[1]):
            g[i] = (self.target(self.x + h * Pk[:, i], z) - fx)
        g.div_(h)
        return Pk.mv(g)

        
    def step(self):
        Pk = self.P()
        h = max(self.h(self.k), self.min_val)
        alpha = max(self.alpha(self.k), self.min_val)
        z = self.sampling_fun() if self.sampling_fun is not None else None
        fx = self.target(self.x, z)
        g = self._approx_grad(fx, Pk, h, z=z) #Pk.matmul((self.target(self.x + h * Pk.T) - fx).div(h)).flatten()
#        exit()
#        print("[--] g: ", g)
#        print(self.x, alpha)
        self.x.add_(g, alpha=-alpha)
 #       print(self.x)
 #       exit()
        self.f_evals += (self.l + 1)
        self.k += 1
        return self.x.clone()
    
    
    