
import torch
import numpy as np

from typing import Callable, Dict, Tuple, Any
from fdopt.direction_generators import DirectionGenerator

from concurrent.futures import ProcessPoolExecutor as PPE, ThreadPoolExecutor as TPE

class Optimizer:
    
    def __init__(self, P : DirectionGenerator, parallel_execution : Tuple[str, int] | None = None) -> None:
        self.P = P
        self.executor_cls = None
        if parallel_execution is not None:
            if parallel_execution[0] == 'process':
                self.executor_cls = PPE
            elif parallel_execution[0] == 'thread':
                self.executor_cls = TPE
        if self.executor_cls is not None:
            self.num_workers = parallel_execution[1]
        
    @property
    def d(self) -> int:
        return self.P.d
    
    @property
    def l(self) -> int:
        return self.P.l
    
    def _build_result(self, x : np.ndarray, fx : float, trace = None):
        result = {'x' : x, 'fx' : fx}
        if trace is not None:
            result['trace'] = trace
        return result
    
    def optimize(self, f : Callable, x_0 : np.ndarray, T : int = 100) -> Dict[str, Any]:
        raise NotImplementedError("This method is implemented in subclasses!")
