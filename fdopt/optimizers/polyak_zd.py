import torch

import numpy as np
from typing import Callable, Dict, Iterable
from fdopt.grad_estimators.grad_estimators import GradEstimator
from fdopt.direction_generators.generator import DirectionGenerator

class PolyakZD:

    def __init__(self, target, g : GradEstimator, P: DirectionGenerator,h: float, gamma_0 : float = 1.0) -> None:
        self.f = target
        self.g = g
        self.h =h
        self.gamma_0 = gamma_0
        self.P = P
    
    def optimize(self, x_0: np.ndarray, f_star, T: int | None = 100, eps=1e-3, verbose = False) -> Dict:
        x_k = x_0.clone()
        f_k = self.f(x_k)
        g_k = None
        trace = [(x_k, f_k)] #if return_trace else None
        k = 0
        num_evals = 0
        gamma =self.gamma_0
        while (T is None or k < T): 
            if  abs(f_k - f_star)<eps :
                break
            h_k = self.h(k)
            P_k = self.P()
            g_k = self.g(self.f, x = x_k, h=h_k, P=P_k)
            g_k_norm = torch.linalg.norm(g_k, ord=2).square()
            num_evals += P_k.shape[0] + 1
            gamma = self.gamma_0 * ((f_k - f_star) / g_k_norm)
            x_k = x_k - gamma * g_k
            f_k = self.f(x_k)
            if verbose:
                print("[--] f_k = {}".format(f_k.cpu().item()))
            trace.append((x_k, f_k)) 
            k += 1
        return  num_evals
