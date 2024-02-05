import torch

import numpy as np
from typing import Callable, Dict, Iterable
from fdopt.grad_estimators.grad_estimators import GradEstimator
from fdopt.direction_generators.generator import DirectionGenerator

class LSZD:

    def __init__(self, target, g : GradEstimator, P: DirectionGenerator,h: float, gamma_0 : float = 1.0, theta : float = 0.5, rho : float = 1.0, armijo_constant : float = 0.8,  max_gamma = 10.0, min_gamma=1e-5) -> None:
        assert 0 < theta < 1 and rho >= 1
        self.f = target
        self.g = g
        self.h =h
        self.gamma_0 = gamma_0
        self.P = P
        self.theta = theta # contraction factor
        self.rho = rho # expansion factor
        self.armijo_constant = armijo_constant
        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
    
    def optimize(self, x_0: np.ndarray, f_star, T: int | None = 100, eps=1e-3, verbose = False, return_trace : bool = False) -> Dict:
        x_k = x_0.clone()
        f_0 = f_k = self.f(x_k)
        g_k = None
        trace = [(x_k, f_k)] if return_trace else None
        k = 0
        num_evals = 0
        gamma =self.gamma_0
        while (T is None or k < T): 
            if  abs(f_k - f_star)<eps :
                break
            h_k = self.h(k)
            #    def __call__(self, target, x, h, P):
            P_k = self.P()
            g_k = self.g(self.f, x = x_k, h=h_k, P=P_k)
            g_k_norm = torch.linalg.norm(g_k, ord=2).square()
            num_evals += P_k.shape[0] + 1
            while self.f(x_k - gamma * g_k) >= f_k - gamma * self.armijo_constant * g_k_norm:
                num_evals += 1
                print("\t[LS] f = {}\tgamma = {}".format(self.f(x_k - gamma * g_k).cpu().item(), gamma))
                gamma = gamma * self.theta
                if gamma <= self.min_gamma:
                    break
            x_k = x_k - gamma * g_k
            gamma = min(gamma * self.rho, self.max_gamma)
            f_k = self.f(x_k)
            if verbose:
                print("[--] f_k = {}".format(f_k.cpu().item()))
            if return_trace:
                trace.append((x_k, f_k)) 
            k += 1
        return  num_evals
