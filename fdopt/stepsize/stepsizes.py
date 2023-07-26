import torch
from typing import Callable, Union

    
class Stepsize:
    
    def __init__(self, gamma_0 : float, law : Callable[[int], float]) -> None:
        self.gamma_0 = gamma_0
        self.law = law
        
    def __call__(self, k : int, fx : Union[torch.Tensor, float] = None, gk : torch.Tensor = None ) -> float:
        return self.gamma_0 * self.law(k)
    
    def update(self, x : torch.Tensor, fx : Union[torch.Tensor, float]) -> None:
        pass
    
    def reset(self):
        pass


class ConstantStepsize(Stepsize):
    
    def __init__(self, gamma_0 : float) -> None:
        super().__init__(gamma_0 = gamma_0, law=None)
        
    def __call__(self, k : int, fx : Union[torch.Tensor, float] = None, gk : torch.Tensor = None ) -> float:
        return self.gamma_0

class PolyakStepsize(Stepsize):
    
    
    def __init__(self, gamma_0 : float, f_min : float = None, eps : Callable[[int], float] | None = None) -> None:
        assert f_min is not None or eps is not None
        self.gamma_0 = gamma_0
        self.f_min = f_min
        self.eps = eps
#        self.__call__ = self._full_polyak if eps is None else self._app_polyak

    def __call__(self, k: int, fx: torch.Tensor | float = None, gk: torch.Tensor = None) -> float:
        if self.eps is None:
            return self._full_polyak(k, fx, gk,)
        return self._app_polyak(k, fx, gk)

    def _full_polyak(self, k : int, fx : Union[torch.Tensor, float] = None, gk : torch.Tensor = None ) -> float:
        sq_norm = torch.linalg.norm(gk, ord=2)#.square()
#        print(f"\t[--] Norm: {sq_norm}")
        return self.gamma_0 * ((fx - self.f_min)/sq_norm)

        
    def _app_polyak(self, k : int, fx : Union[torch.Tensor, float] = None, gk : torch.Tensor = None ) -> float:
        eps_k = self.eps(k)
        sq_norm = gk.norm().square()
        if self.f_min is None or self.f_min > fx:
            self.f_min = fx

        return self.gamma_0 * ((fx - self.f_min + eps_k) / sq_norm)
        
        
        