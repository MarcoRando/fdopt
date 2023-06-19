import torch

from torch_dct import dct

from typing import Union, Callable, Dict

from fdopt.direction_generators.generator import DirectionGenerator
from fdopt.direction_generators.structured_directions import CoordinateDirections

class GradEstimator:
    
    def __init__(self, d : int, l : int, f : Callable[[torch.Tensor], Union[torch.Tensor, float]], P : DirectionGenerator, normalize : bool = True):
        self.f = f
        self.d = d
        self.l = l
        self.P = P
        self.normalize = normalize
        self.nrm_const = (self.d / self.l) if normalize else 1.0
        
        
    def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
        raise NotImplementedError("Function implemented in subclasses")
    
    def update(self, x : torch.Tensor, g : torch.Tensor):
        pass
        
    def reset(self):
        pass
    
    
class ForwardFD(GradEstimator):
        
    def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
        P = self.P()
        return (self.f(x.add(P, alpha=h)) - fx).mul(P).div(h).sum(dim=0, keepdims=True).mul_(self.nrm_const)
        

class CentralFD(GradEstimator):
        
    def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
        P = self.P()
        return (self.f(x.add(P, alpha=h)) - self.f(x.add(P, alpha=-h))).mul(P).div(2 * h).sum(dim=0, keepdims=True).mul_(self.nrm_const)
        
class SinglePointEstimator(GradEstimator):
    def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
        P = self.P()
        return self.f(x.add(P, alpha=h)).mul(P).div(h).sum(dim=0, keepdims=True).mul_(self.nrm_const)

        
class LinInterpolation(GradEstimator):
    def __init__(self, d : int, l : int, f : Callable[[torch.Tensor], Union[torch.Tensor, float]], P : DirectionGenerator, normalize : bool = True):
        assert P.d == P.l, "Number of directions must be equal to the number of dimensions!"
        super().__init__(d, l, f, P, normalize)
        
    def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
        P = self.P()
        f_vec =(self.f(x.add(P, alpha=h)) - fx).reshape(-1)
        return torch.linalg.solve(h * P, f_vec)
    
class CentralLinInterpolation(LinInterpolation):
        
    def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
        P = self.P()
        f_plus  = (self.f(x.add(P, alpha=h))).reshape(-1)
        f_minus = (self.f(x.add(P, alpha=-h))).reshape(-1)
        g_plus  = torch.linalg.solve(h * P, f_plus) 
        g_minus = torch.linalg.solve(h * P, f_minus)
        return (g_plus - g_minus)/2


class GenGradApprox(GradEstimator):
        
                    
    def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
        P = self.P()        
        alpha = (self.f(x.add(P, alpha=h)) - fx)/h
        return torch.linalg.lstsq(P, alpha)[0]

class CentralGenGradApprox(GradEstimator):
        
                    
    def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
        P = self.P()        
        alpha = (self.f(x.add(P, alpha=h)) - self.f(x.add(P, alpha=-h)))/(2 * h)
        return torch.linalg.lstsq(P, alpha)[0] 


class GenStructGradApprox(GradEstimator):
        
    def __init__(self,  **kwargs : Dict):
        super().__init__(**kwargs)
        self.E = dct(torch.eye(self.l, self.d, dtype = self.P.dtype, device = self.P.device), norm='ortho')
        self.E_pinv = torch.linalg.pinv(self.E)

                    
    def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
        G = self.P()
        V = G @ self.E
        alpha = (self.f(x + h * V) - fx)/h
        return self.E_pinv @ G.T @ alpha


class CentralGenStructGradApprox(GenStructGradApprox):
        
                    
    def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
        G = self.P()
        V = G @ self.E
        alpha = (self.f(x + h * V) - self.f(x - h * V))/(2 * h)
        return self.E_pinv @ G.T @ alpha
