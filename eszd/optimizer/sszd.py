import torch
from eszd.optimizer.optimizer import Optimizer
from eszd.directions.directions import DirectionMatrix

class SSZD(Optimizer):
    
    def __init__(self, target, x0, P : DirectionMatrix, alpha, h):
        super().__init__(target, x0)
        self.P = P
        self.alpha = alpha
        self.h = h
        self.t = 1


    def step(self):
        fx = self.target(self.x)
        Pk = self.P()
        alpha_k = self.alpha(self.t)
        h_k = self.h(self.t)
        Fx = target(self.x + h_k * P_k.T)
        self.x -= alpha_k * (Pk @ ((Fx - fx) / h_k))
        self.t+=1
        return self.x

