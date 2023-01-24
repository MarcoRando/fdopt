import torch
import numpy as np
from eszd.optimizer.optimizer import Optimizer
from eszd.directions.directions import DirectionMatrix
import time
class ESSZD(Optimizer):
    
    def __init__(self, P : DirectionMatrix, Lk, h, theta = 1.0, sampling_procedure = None):
        super().__init__("SSZD")
        self.P = P
        self.Lk = Lk
        self.theta = theta
        self.h = h
        self.sampling_procedure = sampling_procedure
        self.t = 1

    @property
    def l(self):
        return self.P.l


    def optimize(self, target, x, T = 1000, eps = 1e-5):
        # T iterations => T * (l + 1) function evaluations 
        mean_g_time = []
        trace = [target(x)]
        Lks = []
        print("[--] t: {}\tfx: {}".format(0, target(x)))

        alpha_k = 1/self.Lk
        prev_g, prev_g_norm, g_norm = None, None, None
        prev_alpha = alpha_k
        evals_per_iter= [self.l]
        for t in range(1, T + 1):
            z = None if self.sampling_procedure is None else self.sampling_procedure()
            fx = target(x, z)
            Pk = self.P()
#            alpha_k = self.alpha
            h_k = self.h(t)
            
            it_time = time.time()
            Fx = target(x + h_k * Pk.T, z) - fx
            Fx /=h_k
            g = Pk @ Fx            
            it_time = time.time() - it_time
            mean_g_time.append(it_time)
            nevals = self.l + 1
            g_norm = torch.linalg.norm(g)
            if t > 1:
                diff_norm = torch.linalg.norm(g - prev_g)
                new_Lk = diff_norm / (prev_alpha * prev_g_norm)
                Lks.append(1/new_Lk.cpu().numpy())
                alpha_k = np.mean(Lks)
#                self.Lk = min(self.Lk, new_Lk)
#                nevals +=1
#                print("--")
#                print(fx - alpha_k * self.theta * g_norm)
#                print(target(x - alpha_k * g, z))
#                print("--")
                
#                while target(x - alpha_k * g, z) > fx - alpha_k * self.theta * g_norm:
#                    self.Lk *= 2
#                    alpha_k = 1/self.Lk
#                    if alpha_k < 1e-5:
#                        alpha_k = 1e-5
#                        break
#                    print("\t[--] alpha: {}".format(alpha_k))
#                    nevals +=1
   #         if t == 3:
   #             exit()
            x = x - alpha_k * g
            
            prev_alpha = alpha_k
            prev_g = g
            prev_g_norm = g_norm
            trace.append(target(x))
            print("[--] t: {}/{}\tfx: {}".format(t, T, target(x)))

            evals_per_iter.append(nevals)
        return x, mean_g_time, trace, evals_per_iter

