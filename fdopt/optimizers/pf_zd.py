import torch
from numpy import mean
from fdopt.optimizers.opt import Optimizer
from typing import Any, Callable, Dict, Tuple
from fdopt.stepsize.stepsizes import ConstantStepsize
from fdopt.grad_estimators.grad_estimators import GradEstimator


class PFZD(Optimizer):
    
    def __init__(self, optimizer : Optimizer) -> None:
        self.optimizer = optimizer
        
    def optimize(self, x_0: torch.Tensor, T: int = 100, num_split : int = 50, gamma_space : Tuple = (1e-7, 1e-1), return_trace : bool = False) -> Dict[str, Any]:
        budget_per_run = T // num_split
        print("num iters per split", budget_per_run)
        assert budget_per_run > 0
        best_obs = (None, None)
        gmin, gmax = gamma_space
        c_val = f0 = self.optimizer.f(x_0)
        trace = [(x_0, f0)]
        for _ in range(num_split):
            gmean = mean([gmin, gmax])
            self.optimizer.reset()
            self.optimizer.gamma = ConstantStepsize(gmean)
            ris = self.optimizer.optimize( x_0 = x_0, T = budget_per_run, convergence_criterion=None, return_trace = return_trace)
            fx = ris['fx'].item()
            if return_trace:
                trace += ris['trace']
            if fx != fx or fx > f0:
                gmax = gmean
            elif fx < c_val:
                gmax = gmean
                c_val = fx
            else:
                gmin = gmean
            if fx == fx and (best_obs[1] is None or fx < best_obs[1]):
                best_obs = (ris, fx)
        return best_obs[0]
            
    def reset(self):
        self.optimizer.reset()
    
    