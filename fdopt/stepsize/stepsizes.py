import numpy as np
from typing import Callable

    
class Stepsize:
    
    def __init__(self, gamma_0 : float, law : Callable[[int], float]) -> None:
        self.gamma_0 = gamma_0
        self.law = law
        
    def __call__(self, k : int, fx : float = None, gk : np.ndarray = None ) -> float:
        return self.gamma_0 * self.law(k)


class ConstantStepsize(Stepsize):
    
    def __init__(self, gamma_0 : float) -> None:
        super().__init__(gamma_0 = gamma_0, law=None)
        
    def __call__(self, k : int, fx : float = None, gk : np.ndarray = None ) -> float:
        return self.gamma_0

class PolyakStepsize(Stepsize):
    
    
    def __init__(self, gamma_0 : float, f_min : float = None, eps : Callable[[int], float] = None) -> None:
        assert f_min is not None or eps is not None
        self.gamma_0 = gamma_0
        self.f_min = f_min
        self.eps = eps
        self.__call__ = self._full_polyak if eps is None else self._app_polyak

    def _full_polyak(self, k : int, fx : float = None, gk : np.ndarray = None ) -> float:
        return self.gamma_0 * ((fx - self.f_min)/np.square(np.linalg.norm(gk)))

        
    def _app_polyak(self, k : int, fx : float = None, gk : np.ndarray = None ) -> float:
        eps_k = self.eps(k)
        if self.f_min is None or self.f_min > fx:
            self.f_min = fx

        return self.gamma_0 * ((fx - self.f_min + eps_k) / np.square(np.linalg.norm(gk)))
        
        
        