import torch

import numpy as np
from typing import Callable, Dict, Iterable
from fdopt.grad_estimators.grad_estimators import GradEstimator
from fdopt.direction_generators.generator import DirectionGenerator

class PFZD:

    def __init__(self, target, g : GradEstimator, P: DirectionGenerator,h: float, num_splits = 3, max_gamma = 10.0, min_gamma=1e-5) -> None:
        self.f = target
        self.g = g
        self.h =h
        self.num_splits = num_splits
        self.P = P
        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
    
    def _zd_descent(self, x0, f_star, T, gamma, eps):
        x_k = x0.clone()
        f_0 = f_k = self.f(x_k)
        trace = []
        num_evals = 0
        converged = False
        for k in range(T):    
            if  abs(f_k - f_star)<eps :
                converged = True
                break
            h_k = self.h(k)
            P_k = self.P()
            g_k = self.g(self.f, x = x_k, h=h_k, P=P_k)
            num_evals += P_k.shape[0] + 1
            x_k = x_k - gamma * g_k
            f_k = self.f(x_k)
            # if verbose:
            #     print("[--] f_k = {}".format(f_k.cpu().item()))
            trace.append((x_k, f_k)) 
        return trace, num_evals, converged

            
    
    def optimize(self, x_0: np.ndarray, f_star, T: int | None = 100, eps=1e-3, verbose = False) -> Dict:
        x_k = x_0.clone()
        f_k = self.f(x_k)
        trace = [(x_k, f_k)] 
        num_evals = 0
        T_split = T // self.num_splits
        best_num_evals = np.inf
        max_gamma, min_gamma =self.max_gamma, self.min_gamma
        best_gamma = (max_gamma + min_gamma) / 2
        for i in range(self.num_splits + 1):
            gamma = (max_gamma + min_gamma) / 2
            split_trace, split_evals, split_converged = self._zd_descent(x_k, f_star, T_split, gamma, eps)
            num_evals += split_evals
            trace += split_trace
            if split_converged and split_evals <= best_num_evals:
                best_gamma = min_gamma = gamma
                print("\tConverged with gamma = {}".format(gamma))
                break
            else:
                max_gamma = gamma
        return num_evals
