import torch
from eszd.optimizer.optimizer import Optimizer
from eszd.directions.directions import DirectionMatrix
import time
class SSZD(Optimizer):
    
    def __init__(self, P : DirectionMatrix, alpha, h, sampling_procedure = None):
        super().__init__("SSZD")
        self.P = P
        self.alpha = alpha
        self.h = h
        self.sampling_procedure = sampling_procedure
        self.t = 1

    @property
    def l(self):
        return self.P.l


    def optimize(self, target, x, T = 1000, eps = 1e-5):
        mean_g_time = []
        trace = [target(x)]
        for t in range(1, T+1):
            z = None if self.sampling_procedure is None else self.sampling_procedure()
            fx = target(x, z)
            print("[--] t: {}\tfx: {}".format(t, fx))
            Pk = self.P()
            alpha_k = self.alpha(t)
            h_k = self.h(t)
            
            it_time = time.time()
            Fx = target(x + h_k * Pk.T, z) - fx
            Fx /=h_k
            g = Pk @ Fx
            it_time = time.time() - it_time
            mean_g_time.append(it_time)
#            if g.T @ g <= eps:
#                return x, mean_g_time
            x = x - alpha_k * g
            trace.append(target(x))
        return x, mean_g_time, trace

