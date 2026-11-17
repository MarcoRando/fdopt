from collections.abc import Callable

import torch


class GradEstimator:
    
    def __init__(self, norm_const : float = 1.0):
        self.nrm_const = norm_const
                    
    def __call__(self, f : Callable[[torch.Tensor], torch.Tensor], x : torch.Tensor, P : torch.Tensor, h : float):
        raise NotImplementedError("Method implemented in subclasses")
                

class ForwardFD(GradEstimator):
        
    def __call__(self, f : Callable[[torch.Tensor], torch.Tensor], x : torch.Tensor,  P : torch.Tensor, h : float):        
        x_next = x + h * P
        diff = f(x_next) - f(x.repeat_interleave(P.shape[0], dim=0))
        diff.div_(h)
        return (diff.reshape(P.shape[0], -1) * P).sum(dim=0, keepdims=True).mul(self.nrm_const)        

