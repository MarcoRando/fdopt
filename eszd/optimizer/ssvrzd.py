import time
import torch
from eszd.optimizer.optimizer import Optimizer
from eszd.directions.directions import DirectionMatrix

class SSVRZD(Optimizer):
    
    def __init__(self, 
                 P : DirectionMatrix, 
                 alpha, h,
                 sampling_procedure,
                 m = 10,
                 update_P = 'always'
                 ):
        self.P = P
        self.update_P = update_P
        self.alpha = alpha
        self.h = h
        self.sampling_procedure = sampling_procedure
        self.t = 1
        self.m = m
        
    @property
    def l(self):
        return self.P.l

    def _approx_grad(self, target, x, h, P, fx, z = None):
        Fx = target(x + h * P.T, z) - fx
        Fx /=h
        return P @ Fx



    def optimize(self, target, x, T = 1000):
        mean_g_time = []
        trace = []
        for t in range(1, T+1):
            Pk = self.P()
            alpha_k = self.alpha(t)
            h_k = self.h(t)
            x_j = x.clone()
            trace.append(x)
            outer_it = time.time()
            G_t = self._approx_grad(target, x, h_k, Pk, target)
            for j in range(self.m):
                z_j = self.sampling_procedure()
                P_j = self.P() if j % self.update_P == 0 else Pk
                g_t = self._approx_grad(target, x, h_k, P_j, target, z_j)
                g_j = self._approx_grad(target, x_j, h_k, P_j, target, z_j)
                x_j = x_j - alpha_k *(G_t + (g_j - g_t))
            outer_it = time.time() - outer_it
            mean_g_time.append(outer_it)
            x = x_j
#            fx = target(x, z)
 #           print("[--] t: {}\tfx: {}".format(t, fx))          
#            it_time = time.time()
#            Fx = target(x + h_k * Pk.T, z) - fx
#            Fx /=h_k
#            g = Pk @ Fx
#            it_time = time.time() - it_time
#            mean_g_time.append(it_time)
        return x, mean_g_time, trace

