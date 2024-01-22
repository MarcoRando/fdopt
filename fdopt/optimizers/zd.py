import torch

from fdopt.optimizers.opt import Optimizer
from typing import Callable, Dict, Iterable
from fdopt.stepsize.stepsizes import Stepsize
from fdopt.grad_estimators.grad_estimators import GradEstimator


class ZD(Optimizer):
    
    def __init__(self, sigma = lambda t : 0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sigma = sigma

    def _approximate_grad(self, x):
        P = self.P()
        h = self.h(self.t)
        fx = self.target(x)
        g = torch.zeros_like(x, dtype=self.P.dtype, device=self.P.device)
        for i in range(P.shape[0]):
            g += ((self.target(x + h * P[i]) - fx) / h) * P[i]
        return g, fx

    def step(self):            
        g, fx = self._approximate_grad(self.x)
        x_new = self.x - self.gamma * g
        fx_new = self.target(x_new)
        g_norm = torch.linalg.norm(g, ord=2).square()
        num_evals = self.P.l + 2
        while fx_new > fx -  self.gamma * self.armijo_constant * g_norm:
            self.gamma = max(self.gamma * self.theta, self.min_gamma)
            x_new = self.x - self.gamma * g
            if self.gamma == self.min_gamma:
                break
            fx_new = self.target(x_new)
            num_evals += 1
        err = torch.randn(x_new.shape, generator=self.generator, dtype = self.P.dtype)
        self.x = x_new + self.sigma(self.t) * err
        self.gamma = min(self.gamma * self.rho, self.max_gamma)
        self.t += 1
        return self.target(self.x), num_evals
        
    def __str__(self) -> str:
        return str(self.g)
    
    def reset(self):
        return super().reset()