import torch
import tqdm

from typing import Callable, Any
from fdopt.direction_generators import DirectionGenerator
from fdopt.grad_estimators import GradEstimator

class FDOpt:
        
    def __init__(self, 
                 P : DirectionGenerator,
                 g : GradEstimator,
                 gamma : Callable[[Any], float] | float, 
                 h : Callable[[Any], float] | float, 
                 seed : int = 121314) -> None:
        self.P = P
        self.g = g
        self.gamma = gamma if isinstance(gamma, Callable) else lambda t : gamma
        self.h = h if isinstance(h, Callable) else lambda t : h
#        self.generator = torch.Generator(P.device).manual_seed(seed)
        
    def optimize(self, f : Callable, x_0 : torch.Tensor, T : int):
        x_k = x_0.clone()
        f_k = f(x_k)
        f_values = [f_k]
        iterator = tqdm.tqdm(range(T))
        for k in iterator:
            gamma_k, h_k = self.gamma(k), self.h(k)
            P_k = self.P()
            g = self.g(f, x_k, f_k, P_k, h_k)
#            print(x_k, g)
            x_k = x_k - gamma_k * g
            f_k = f(x_k)
            f_values.append(f_k.cpu().item())
            iterator.set_postfix({'f(x_k)' : f_values[-1]})
        return x_k, f_values

