import torch
from eszd.optimizer.optimizer import Optimizer
from eszd.directions.directions import DirectionMatrix

class SSZD(Optimizer):
    
    def __init__(self, P : DirectionMatrix, alpha, h):
        self.P = P
        self.alpha = alpha
        self.h = h
        

    def optimize(target, x : torch.Tensor, T : int, eps : float = 1e-10):
        for t in range(T):
            fx = target(x)
            Pk = self.P()
          #  Fx = 
            
