import torch 
from .opt import Optimizer

class SZD(Optimizer):
    
    def __approx_grad__(self, fun, x, h, *args):
        P = self.P()
        return (fun(x + h * P, *args) - fun(x, *args)).div_(h).sum(dim = 0, keepdim=True)
    
    
    def step(self, fun, x, *args):
        alpha = self.alpha()
        h = self.h()
        d = self.__approx_grad__(fun, x, h, *args)
        return x - alpha * d