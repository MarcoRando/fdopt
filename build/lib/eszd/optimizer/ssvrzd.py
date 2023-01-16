import torch
from eszd.optimizer.optimizer import Optimizer
from eszd.directions.directions import DirectionMatrix

class ESSVRZD(Optimizer):
    
    def __init__(self, target, 
                 x0, 
                 P : DirectionMatrix, 
                 alpha, 
                 h,
                 m = 10
                 ):
        super().__init__(target, x0)
        self.P = P
        self.alpha = alpha
        self.best_g = None
        self.h = h
        self.t = 1
        self.m = m


    def step(self):
        fx = self.target(self.x)
        Pk = self.P()
        Fx = target(self.x + h_k * P_k.T)
        g = (Pk @ ((Fx - fx) / h_k))
        if self.best_g is None:            
            alpha_k = self.alpha
        self.x -= alpha_k * (Pk @ ((Fx - fx) / h_k))
        self.t+=1
        return self.x

