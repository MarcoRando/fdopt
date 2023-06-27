import torch

from fdopt.optimizers.opt import Optimizer
from typing import Callable, Dict, Iterable
from fdopt.stepsize.stepsizes import Stepsize
from fdopt.grad_estimators.grad_estimators import GradEstimator


class ZD(Optimizer):

    def __init__(self, g : GradEstimator, gamma : Stepsize, h: Stepsize) -> None:
        self.g = g
        self.gamma = gamma
        self.h = h

    @property
    def f(self):
        return self.g.f        
    
    def optimize(self, x_0: torch.Tensor, T: int | None = 100, convergence_criterion : Callable | None = None, return_trace : bool = False) -> Dict:
        assert T is not None or convergence_criterion is not None, "You have to pass a maximum number of iteration (T) or a convergence criterion!"
        x_k = x_0.clone()
        f_0 = f_k = self.f(x_k)
        g_k = None
        trace = [(x_k, f_k)] if return_trace else None
        k = 0
        while (T is None or k < T): 
            if (convergence_criterion is None or convergence_criterion(k, g_k, f_k, f_0)):
                break
            h_k = self.h(k)
            g_k = self.g(x = x_k, fx=f_k, h=h_k)
            gamma_k = self.gamma(k=k, fx=f_k, gk=g_k)
            x_k = x_k - gamma_k * g_k
            f_k = self.f(x_k)
            if return_trace:
                trace.append((x_k, f_k)) 
            self.g.update(x_k, g_k)
            self.gamma.update(x_k, f_k)
            k += 1
        converged = None if convergence_criterion is None else convergence_criterion(k, g_k, f_k, f_0)
        return self._build_result(x_k, f_k, k, trace=trace, converged=converged)
            
    def reset(self):
        self.g.reset()
        self.gamma.reset()
        self.h.reset()