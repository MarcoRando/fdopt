import torch

from typing import Union, Callable, Dict

from fdopt.direction_generators.generator import DirectionGenerator
from fdopt.direction_generators.structured_directions import CoordinateDirections

class GradEstimator:
    
    def __init__(self, norm_const : float = 1.0):
        self.nrm_const = norm_const
                    
    def __call__(self, f : Callable[[torch.Tensor], torch.Tensor], x : torch.Tensor, P : torch.Tensor, h : float):
        raise NotImplementedError("Method implemented in subclasses")
                

class ForwardFD(GradEstimator):
        
    def __call__(self, f : Callable[[torch.Tensor], torch.Tensor], x : torch.Tensor,  P : torch.Tensor, h : float):        
        return ((f(x + h * P) - f(x.repeat_interleave(P.shape[0], dim=0)) ).div(h).reshape(P.shape[0], -1) * P).sum(dim=0, keepdims=True).mul(self.nrm_const)        

class CentralFD(GradEstimator):
        
    def __call__(self, f : Callable[[torch.Tensor], torch.Tensor], x : torch.Tensor,  P : torch.Tensor, h : float):
        return ((f(x + h * P) - f(x - h * P)).div(2*h) * P.T).sum(dim=1, keepdims=True).mul(self.nrm_const).T
        
class OptFFD(GradEstimator):
    
    def __call__(self, f : Callable[[torch.Tensor], torch.Tensor], x : torch.Tensor,  P : torch.Tensor, h : float):
        alpha = (f(x + h * P) - f(x.repeat_interleave(P.shape[0], dim=0)))/h
        return torch.linalg.lstsq( P, alpha)[0].reshape(1, -1)

class OptCFD(GradEstimator):
    
    def __call__(self, f : Callable[[torch.Tensor], torch.Tensor], x : torch.Tensor,  P : torch.Tensor, h : float):
#        f_plus = f(x + h * P).reshape(-1) /h
#        f_min = f(x - h * P).reshape(-1)/h
        f_plus = (f(x + h * P).reshape(-1) -  f(x - h * P).reshape(-1))/(2 * h)
        return torch.linalg.lstsq(P, f_plus)[0].reshape(1, -1)
#        g_plus = torch.linalg.lstsq(P, f_plus)[0].reshape(1, -1)
#        g_min = torch.linalg.lstsq(P,  f_min)[0].reshape(1, -1)
#        return (g_plus - g_min) / 2

# class CentralLinInterpolation(LinInterpolation):
        
#     def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
#         P = self.P()
#         f_plus  = (self.f(x.add(P, alpha=h))).reshape(-1)
#         f_minus = (self.f(x.add(P, alpha=-h))).reshape(-1)
#         g_plus  = torch.linalg.solve(h * P, f_plus) 
#         g_minus = torch.linalg.solve(h * P, f_minus)
#         return (g_plus - g_minus)/2


        
# class SinglePointEstimator(GradEstimator):
#     def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
#         P = self.P()
#         return self.f(x.add(P, alpha=h)).mul(P).div(h).sum(dim=0, keepdims=True).mul_(self.nrm_const)
#     def __str__(self):
#         return f"SPE_{self.P}"

        
# class LinInterpolation(GradEstimator):
#     def __init__(self, d : int, l : int, f : Callable[[torch.Tensor], Union[torch.Tensor, float]], P : DirectionGenerator, norm_const : float = 1.0):
#         assert P.d == P.l, "Number of directions must be equal to the number of dimensions!"
#         super().__init__(d, l, f, P, norm_const)
        
#     def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
#         P = self.P()
#         f_vec =(self.f(x.add(P, alpha=h)) - fx).reshape(-1)
#         return torch.linalg.solve(h * P, f_vec)
    
#     def __str__(self):
#         return f"LI_{self.P}"

    
# class CentralLinInterpolation(LinInterpolation):
        
#     def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
#         P = self.P()
#         f_plus  = (self.f(x.add(P, alpha=h))).reshape(-1)
#         f_minus = (self.f(x.add(P, alpha=-h))).reshape(-1)
#         g_plus  = torch.linalg.solve(h * P, f_plus) 
#         g_minus = torch.linalg.solve(h * P, f_minus)
#         return (g_plus - g_minus)/2

#     def __str__(self):
#         return f"CLI_{self.P}"


# class GenGradApprox(GradEstimator):
        
                    
#     def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
#         P = self.P()        
#         alpha = (self.f(x.add(P, alpha=h)) - fx)/h
#         return torch.linalg.lstsq(P, alpha)[0].reshape(1, -1)
    
#     def __str__(self):
#         return f"GGA_{self.P}"

# class CentralGenGradApprox(GradEstimator):
        
                    
#     def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
#         P = self.P()        
#         alpha = (self.f(x.add(P, alpha=h)) - self.f(x.add(P, alpha=-h)))/(2 * h)
#         return torch.linalg.lstsq(P, alpha)[0] 
#     def __str__(self):
#         return f"CGGA_{self.P}"


# class GenStructGradApprox(GradEstimator):
        
#     def __init__(self,  **kwargs : Dict):
#         super().__init__(**kwargs)
#         self.E = dct(torch.eye(self.l, self.d, dtype = self.P.dtype, device = self.P.device), norm='ortho')
#         self.E_pinv = torch.linalg.pinv(self.E)

                    
#     def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
#         G = self.P()
#         V = G @ self.E
#         alpha = ((self.f(x + h * V) - fx)/h)#.reshape(-1)
#         return (self.E_pinv @ G.T @ alpha).T
    
#     def __str__(self):
#         return f"GSGA_{self.P}"


# class CentralGenStructGradApprox(GenStructGradApprox):
        
                    
#     def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
#         G = self.P()
#         V = G @ self.E
#         alpha = (self.f(x + h * V) - self.f(x - h * V))/(2 * h)
#         return (self.E_pinv @ G.T @ alpha).T
#     def __str__(self):
#         return f"CGSGA_{self.P}"


# class EfficientGSGA(GradEstimator):
#     def __init__(self, fixed_matrix_mode : str = 'gaussian', eps : float = 1e-3, in_constant = 7.0, verbose : bool = False, **kwargs : Dict):
#         super().__init__(**kwargs)        
#         self.E = self._generated_fixed_matrix(fixed_matrix_mode)
#         self.EtE = self.E.T.matmul(self.E)
#         self.L = torch.linalg.norm(self.EtE)
#         self.eps = eps
#         self.verbose = verbose
#         self.in_constant = in_constant
# #        self.E_pinv = torch.linalg.pinv(self.E)

#     def _generated_fixed_matrix(self, mode):
#         if mode == 'identity':
#             return dct(torch.eye(self.l, self.d, dtype = self.P.dtype, device = self.P.device), norm='ortho')
#         Q = torch.randn((self.d, self.l), generator=self.P.generator, dtype=self.P.dtype)
#         if mode == 'gaussian':
#             return Q.T
#         elif mode == 'spherical':
#             Q /= torch.linalg.norm(Q, ord=2, dim=0)
#             return Q.T
#         elif mode == 'qr':
#             G, _ = torch.linalg.qr(Q)
#             return G.T
#         raise ValueError("Unknown mode!")                    
    
#     def _grad(self, g, b):
#         return self.EtE.mv(g) - b     
    
#     def __call__(self, x : torch.Tensor, fx : Union[torch.Tensor, float], h : float):
#         G = self.P()
#         V = G @ self.E
#         alpha = ((self.f(x + h * V) - fx)/h)#.reshape(-1)
#         b = (V.T @ alpha)#.mv(alpha)
#         return torch.linalg.lstsq(self.EtE, b)[0].T
        
# #        g = torch.ones(self.d, dtype = self.P.dtype, device = self.P.device)
# ##        y = g.clone()
# #        grad = self._grad(g, b)
# ##        c = 1
# #        while grad.norm() > self.eps:
# ##            print(f"[c={c}] |grad|: {torch.linalg.norm(grad)}")
# #            g = g - (1/self.L)*grad   
# #            grad = self._grad(g, b)
# ##            c += 1
# #        return g
    
#     def __str__(self):
#         return f"EGSGA_{self.P}"