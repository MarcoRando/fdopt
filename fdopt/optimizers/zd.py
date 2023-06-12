import torch
import numpy as np

from fdopt.optimizers.opt import Optimizer
from typing import Callable, Dict, Iterable
from fdopt.stepsize.standard_stepsizes import Stepsize

from concurrent.futures import ProcessPoolExecutor as PPE, ThreadPoolExecutor as TPE


class ZD(Optimizer):

    def __init__(self, gamma : Stepsize, h: Stepsize, g_estimator : str = 'forward', **kwargs) -> None:
        assert g_estimator in ['forward', 'single', 'central']
        super().__init__(**kwargs)
        self._approx_grad = getattr(self, '_compute_{}_fd'.format(g_estimator))
        self.gamma = gamma
        self.h = h

    def _compute_forward_fd(self, f : Callable, x : np.ndarray, h : float, P : np.ndarray, values : Iterable, executor: PPE | TPE | None = None):
        g = np.zeros(self.d, dtype=self.P.dtype)
        fx = f(x) if executor is None else executor.submit(f, x).result()
        for i, f_next in enumerate(values):
            g +=  (f_next - fx) * P[i] 
        return (self.d/self.l) * g/h

    def _compute_central_fd(self, f : Callable, x : np.ndarray, h : float, P : np.ndarray, values : Iterable, executor: PPE | TPE | None = None):
        g = np.zeros(self.d, dtype=self.P.dtype)
        near_points = [x - h * P[i] for i in range(self.l)]
        left_values = map(f, near_points) if executor is None else executor.map(f, near_points)
        for i, (f_plus, f_minus) in enumerate(zip(values, left_values)):
            g +=  (f_plus - f_minus) * P[i] 
        return (self.d/self.l) * (g/(2*h))
  
    def _compute_single_fd(self, f : Callable, x : np.ndarray, h : float, P : np.ndarray, values : Iterable, executor: PPE | TPE | None = None):
        g = np.zeros(self.d, dtype=self.P.dtype)
        for i, f_next in enumerate(values):
            g +=  (f_next * P[i]) 
        return (self.d/self.l) * g / h 
        
    
    def optimize(self, f: Callable, x_0: np.ndarray, T: int = 100, return_trace : bool = False) -> Dict:
        x = x_0.copy()
        trace = [(x, f(x))] if return_trace else None

        executor = self.executor_cls(max_workers = self.num_workers) if self.executor_cls is not None else None
        for k in range(T):
            h = self.h(k)
            P = self.P()
            near_points = [x + h * P[i] for i in range(self.l)]
            values = map(f, near_points) if executor is None else executor.map(f, near_points)
            g = self._approx_grad(f, x, h, P, values=values)
            gamma = self.gamma(k)
            x = x - gamma * g
            if return_trace:
                trace.append((x, f(x))) 
        if executor is not None:
            executor.shutdown()
        return self._build_result(x, f(x), trace=trace)
            