import torch

class FDSurrogate:
    
    def __init__(self, vectorized = False) -> None:
        self.vectorized = vectorized
        
    def __call__(self, target, x, h, P):
        pass
    
class ForwardFD(FDSurrogate):
    
    def __call__(self, target, x, h, P):
        if self.vectorized:
            return (P.T.mv(target(x + h * P).add(target(x), alpha=-1).div_(h))).mul(P.shape[1] / P.shape[0])
        # Non-vectorized version       
        fx = target(x) 
        g = torch.zeros_like(x, dtype=P.dtype, device=P.device)
        for i in range(P.shape[0]):
            g += ((target(x + h * P[i]) - fx) / h) * P[i]
        return g
    
class OptimizedForwardFD(FDSurrogate):
    def __call__(self, target, x, h, P):
        y = target(x + h * P).add(target(x), alpha=-1).div_(h)
        return torch.linalg.lstsq((P.shape[0]/P.shape[1]) * P, y)[0]

class CentralFD(FDSurrogate):
    
    
    def __call__(self, target, x, h, P):
#        if self.vectorized:
        return (P.T.mv(target(x + h * P).add(target(x - h * P), alpha=-1).div_(2 * h))).mul(P.shape[1] / P.shape[0])
        # Non-vectorized version
        # g = torch.zeros_like(x, dtype=P.dtype, device=P.device)
        # for i in range(P.shape[0]):
        #     g += ((target(x + h * P[i]) - target(x - h * P[i])) / (2 * h)) * P[i]
        # return (P.shape[1] / P.shape[0]) * g
        