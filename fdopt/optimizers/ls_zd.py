import torch

import numpy as np
from fdopt.optimizers.zd import ZD
from typing import Callable, Dict, Iterable
#from fdopt.stepsize.stepsizes import ConstantStepsize, Stepsize
from fdopt.grad_estimators.grad_estimators import GradEstimator


class LSZD(ZD):

    def __init__(self,  g : GradEstimator, h: float, gamma_0 : float = 1.0, theta : float = 0.5, rho : float = 1.0, armijo_constant : float = 0.8,  max_gamma = 10.0, **kwargs) -> None:
        assert 0 < theta < 1 and rho >= 1
#        gamma = ConstantStepsize(gamma_0=gamma_0)
        super().__init__(g=g, gamma=gamma_0, h = h)
        self.theta = theta # contraction factor
        self.rho = rho # expansion factor
        self.armijo_constant = armijo_constant
        self.max_gamma = max_gamma
        
    def ask(self):
        pass

    def tell(self, X, Y):
        pass
    
    def optimize(self, x_0: np.ndarray, T: int | None = 100, convergence_criterion : Callable | None = None, return_trace : bool = False) -> Dict:
        assert T is not None or convergence_criterion is not None, "You have to pass a maximum number of iteration (T) or a convergence criterion!"
        x_k = x_0.clone()
        f_0 = f_k = self.f(x_k)
        g_k = None
        trace = [(x_k, f_k)] if return_trace else None
        k = 0
        while (T is None or k < T): 
            if (convergence_criterion is not None and convergence_criterion(k, g_k, f_k, f_0)):
                break
            h_k = self.h(k)
            g_k = self.g(x = x_k, fx=f_k, h=h_k)
            gamma_k = self.gamma(k=k, fx=f_k, gk=g_k)
            g_k_norm = torch.linalg.norm(g_k, ord=2).square()
            while self.f(x_k - gamma_k * g_k) >= f_k - gamma_k * self.armijo_constant * g_k_norm:
                gamma_k = gamma_k * self.theta
            x_k = x_k - gamma_k * g_k
            gamma_k = min(gamma_k * self.rho, self.max_gamma)
            self.gamma.gamma_0 = gamma_k
            f_k = self.f(x_k)
            if return_trace:
                trace.append((x_k, f_k)) 
            self.g.update(x_k, g_k)
            self.gamma.update(x_k, f_k)
            k += 1
        converged = None if convergence_criterion is None else convergence_criterion(k, g_k, f_k, f_0)
        return self._build_result(x_k, f_k, k, trace=trace, converged=converged)
