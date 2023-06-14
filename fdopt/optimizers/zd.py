import torch
import numpy as np

from fdopt.optimizers.opt import Optimizer
from typing import Callable, Dict, Iterable
from fdopt.stepsize.standard_stepsizes import Stepsize
from fdopt.grad_estimators.grad_estimators import GradEstimator


class ZD(Optimizer):

    def __init__(self, g : GradEstimator, gamma : Stepsize, h: Stepsize) -> None:
        self.g = g
        self.gamma = gamma
        self.h = h

    @property
    def f(self):
        return self.g.f        
    
    def optimize(self, x_0: torch.Tensor, T: int = 100, return_trace : bool = False) -> Dict:
        x_k = x_0.clone()
        f_k = self.f(x_k)
        trace = [(x_k, f_k)] if return_trace else None

        for k in range(T):
            h_k = self.h(k)
            g_k = self.g(x = x_k, fx=f_k, h=h_k)
            gamma_k = self.gamma(k=k, fx=f_k, gk=g_k)
            x_k = x_k - gamma_k * g_k
            f_k = self.f(x_k)
            if return_trace:
                trace.append((x_k, f_k)) 
            self.g.update(x_k, g_k)
            self.gamma.update(x_k, f_k)
        
        return self._build_result(x_k, f_k, trace=trace)
            
    def reset(self):
        self.g.reset()
        self.gamma.reset()