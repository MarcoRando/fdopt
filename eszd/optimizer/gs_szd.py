import torch

from eszd.optimizer import ESZD

class GSSZD(ESZD):
    
    def __init__(self, 
                 eps : float = 1e-5, 
                 theta : float = 0.5, 
                 init_h : float = 10.0, 
                 max_iters : int = 100,
                 budget : int | None = None, 
                 h_min : float | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        assert 0.0 < theta < 1.0, "Contraction factor must be < 1.0 and > 0.0"
        self.eps = eps
        self.theta = theta
        self.init_h = init_h
        self.max_iters = max_iters
        self.h = init_h
        self.budget = budget
        self.h_min = torch.finfo(self.x.dtype).eps if h_min is None else h_min
        self.t = 1
    def _minimize_fh(self, x):
        k = 1
        g = None
        num_evals = 0
        while k < self.max_iters and (self.budget is None or self.f_evals < self.budget) and  (g is None or g.norm() > self.eps):
            fx = self.target(self.x)
            alpha = self.alpha(k) * self.h
            Pk = self.P()
            g = self._approx_grad(fx, Pk, self.h)
            if self.verbose:
                print("\t[--] |g| = {}".format(g.norm()))
            self.x.add_(g, alpha=-alpha)
            self.f_evals += (self.l + 1)
            num_evals += (self.l + 1)
            k += 1
            self.t += 1
        return num_evals
        
    def step(self):
        n_evals = self._minimize_fh(self.x)
        self.h = max(self.h * self.theta, self.h_min)
        return self.x, n_evals
        
    def reset(self, target = None):
        super().reset(target)
        self.h = init_h
        self.t = 1
        
        