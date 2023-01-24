import torch
from eszd.optimizer.optimizer import Optimizer
from eszd.directions.directions import DirectionMatrix
import time
class SSZD(Optimizer):
    
    def __init__(self, P : DirectionMatrix, alpha, h, sampling_procedure = None, verbose=False):
        super().__init__("SSZD", verbose)
        self.P = P
        self.alpha = alpha
        self.h = h
        self.sampling_procedure = sampling_procedure

    @property
    def l(self):
        return self.P.l


    def optimize(self, target, x, T = 1000, eps = 1e-5):
        # T function evaluations 
        iteration_time = []
        
        trace = [target(x).item()]
        evals_per_iter = []#self.l]
        for t in range(1, T +1):
            it_time = time.time()
            z = None if self.sampling_procedure is None else self.sampling_procedure()
            fx = target(x, z)
            Pk = self.P()
            alpha_k = self.alpha(t)
            h_k = self.h(t)            
            Fx = (target(x + h_k * Pk.T, z) - fx) / h_k
            g = Pk @ Fx
            x = x - alpha_k * g
            it_time = time.time() - it_time
            if self.verbose:
                print("[--] t: {}/{}\tfx: {}".format(t, T, target(x)))
            iteration_time.append(it_time)
            evals_per_iter.append(self.l + 1)
            trace.append(target(x).item())
        return x, iteration_time, trace, evals_per_iter

