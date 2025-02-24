import torch
import tqdm

from typing import Callable, Any
from fdopt.direction_generators import DirectionGenerator
from fdopt.grad_estimators import GradEstimator
from fdopt.stepsize import Stepsize

class FDOpt:
        
    def __init__(self, 
                 P : DirectionGenerator,
                 g : GradEstimator,
                 gamma : Callable[[int, float, torch.Tensor], float] | Stepsize | float, 
                 h : Callable[[Any], float] | float
                 ) -> None:
        self.P = P
        self.g = g
        self.gamma = gamma if isinstance(gamma, Callable) or isinstance(gamma, Stepsize) else lambda t,f,g : gamma
        self.h = h if isinstance(h, Callable) else lambda t : h
        
                
    def optimize(self, f : Callable, x_0 : torch.Tensor, T : int):
        x_k = x_0.clone()
        f_k = f(x_k)
        f_values = [f_k.cpu().item()]
        iterator = tqdm.tqdm(range(T))
        for k in iterator:
            h_k = self.h(k)
            P_k = self.P()
            g_k = self.g(f, x_k, P_k, h_k)
            gamma_k = self.gamma(k, f_k, g_k)
            x_k = x_k - gamma_k * g_k
            f_k = f(x_k)
#            print(f_k.item())
            f_values.append(f_k.cpu().item())
            iterator.set_postfix({'f(x_k)' : f_values[-1]})
        return x_k, f_values



class AccFDOpt:
    
    def __init__(self, 
                 P : DirectionGenerator,
                 g : GradEstimator,
                 gamma : Callable[[int, float, torch.Tensor], float] | Stepsize | float, 
                 alpha : float,
                 h : Callable[[Any], float] | float
                 ) -> None:
        self.P = P
        self.g = g
        self.alpha = alpha
        self.gamma = gamma if isinstance(gamma, Callable) or isinstance(gamma, Stepsize) else lambda t,f,g : gamma
        self.h = h if isinstance(h, Callable) else lambda t : h
        
    def optimize(self, f : Callable, x_0 : torch.Tensor, T : int):
        x_k = x_0.clone()
        f_k = f(x_k)
        f_values = [f_k.cpu().item()]
        iterator = tqdm.tqdm(range(T))
        y_k = x_k.clone()
        t_prev = 1.0
        for k in iterator:
            t_k  = (k + self.alpha) / self.alpha
            beta = (t_prev - 1) / t_k
            h_k = self.h(k)
            P_k = self.P()

            g_k = self.g(f, y_k, P_k, h_k)
            gamma_k = self.gamma(k, f_k, g_k)
            x_next = y_k - gamma_k * g_k
            y_k = x_next + beta * (x_next - x_k)
            x_k = x_next
            t_prev = t_k
            f_values.append(f(x_k).cpu().item())
            iterator.set_postfix({'f(x_k)' : f_values[-1]})
        return x_k, f_values


