import torch
import tqdm
import time

from typing import Callable, Any, Tuple
from fdopt.optimizers.opt import Optimizer
from fdopt.direction_generators import DirectionGenerator
from fdopt.grad_estimators import GradEstimator


class FDOpt:
        
    def __init__(self, 
                 P : DirectionGenerator,
                 g : GradEstimator,
                 gamma : Callable[[int, float, torch.Tensor], float]  | float, 
                 h : Callable[[Any], float] | float
                 ) -> None:
        self.P = P
        self.g = g
        self.gamma = gamma if isinstance(gamma, Callable) else lambda t,f,g : gamma
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
            f_values.append(f_k.cpu().item())
            iterator.set_postfix({'f(x_k)' : f_values[-1] })
        return x_k, f_values



class LSFDOpt:
    
    def __init__(self, 
                P : DirectionGenerator,
                g : GradEstimator,
                gamma_0 : float, 
                h : Callable[[Any], float] | float,
                gamma_minmax : Tuple[float, float] = (1e-5, 1.0),
                theta : float = 0.5,
                rho : float = 2.0,
                c : float = 0.1
                ) -> None:
        self.P = P
        self.g = g
        self.gamma_0 = gamma_0 
        self.h = h if isinstance(h, Callable) else lambda t : h
        self.gamma_minmax = gamma_minmax
        self.theta = theta
        self.rho = rho
        self.c = c
        
                
    def optimize(self, f : Callable, x_0 : torch.Tensor, T : int):
        x_k = x_0.clone()
        f_k = f(x_k)
        f_values = [f_k.cpu().item()]
        num_evals = [1]
        it_times = [0.0]
#        iterator = tqdm.tqdm(range(T))
        gamma = self.gamma_0
        gammas = [gamma]
        k = 0
        while k < T:
            it_time = time.time()
            h_k = self.h(k)
            P_k = self.P()
            g_k = self.g(f, x_k, P_k, h_k)
            sq_norm = g_k.norm(p=2).square()
            k += self.P.l + 1
            num_evals.append(self.P.l + 1)

            while f(x_k - gamma * g_k) >= f_k - gamma * self.c * sq_norm and gamma > self.gamma_minmax[0]:
                gamma = max(self.theta * gamma, self.gamma_minmax[0])
                num_evals[-1] += 1
                k += 1
                if k >= T:
                    return x_k, f_values, num_evals, gammas, it_times
            gammas.append(gamma)
            if f(x_k - gamma * g_k) < f_k - gamma * self.c * sq_norm:
                x_k = x_k - gamma * g_k
                gamma = min(gamma * self.rho, self.gamma_minmax[1])
            it_time = time.time() - it_time
            # else:
            #     gamma = self.gamma_0
            f_k = f(x_k)
            f_values.append(f_k.cpu().item())
            it_times.append(it_time)
            print(f'[--] directions = {self.P}\tk = {k}/{T}\tf(x_k) = {f_k.cpu().item()}\tgamma = {gamma}\ttime = {round(it_time, 4)}')
 #           iterator.set_postfix({'f(x_k)' : f_values[-1] , 'gamma' : gamma})
        return x_k, f_values, num_evals, gammas, it_times
