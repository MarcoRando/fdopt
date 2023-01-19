import time
import torch
from eszd.optimizer.optimizer import Optimizer
from eszd.directions.directions import DirectionMatrix

class SSVRZD(Optimizer):
    
    def __init__(self, 
                 P : DirectionMatrix, 
                 alpha, h,
                 sampling_procedure,
                 update_P = 1
                 ):
        super().__init__("SSVRZD")
        self.P = P
        self.update_P = update_P
        self.alpha = alpha
        self.h = h
        self.sampling_procedure = sampling_procedure
        self.t = 1
        
        
    @property
    def d(self):
        return self.P.d
        
    @property
    def l(self):
        return self.P.l

    def _approx_grad(self, target, x, h, P, fx, z = None):
        Fx = target(x + h * P.T, z) - fx
        Fx /=h
        return P @ Fx



    def optimize(self, target, x, T = 1000, m = 10):
        # T iterations => T * (d + m * (2 * l + 2)) function evaluations
        mean_g_time = []
        fx = target(x)
        trace = [fx]
        evals_per_iter = [self.d + m * (2*self.l + 2)]
        print(T, '->', T * (self.d + m * (2*self.l + 2)) )
        for t in range(1, T + 1):
#            return x, mean_g_time, trace, evals_per_iter

            Pk = self.P()
            alpha_k = self.alpha(t)
            h_k = self.h(t)
            x_j = x.clone()
            print("[--] t: {}/{}\tfx: {}".format(t, T, target(x)))
            outer_it = time.time()
            G_t = self._approx_grad(target, x, h_k, Pk, fx)
            n_eval = self.d
            for j in range(m):
                z_j = self.sampling_procedure()
                P_j = self.P() if (j + 1) % self.update_P == 0 else Pk
                g_t = self._approx_grad(target, x, h_k, P_j, target(x, z_j), z_j)
                g_j = self._approx_grad(target, x_j, h_k, P_j, target(x_j, z_j), z_j)
                x_j = x_j - alpha_k *(g_j - g_t + G_t)
                n_eval += 2 * self.l + 2
            outer_it = time.time() - outer_it
            mean_g_time.append(outer_it)
            evals_per_iter.append(n_eval)
            x = x_j.clone()
            fx = target(x) 
            trace.append(fx)
        return x, mean_g_time, trace, evals_per_iter

