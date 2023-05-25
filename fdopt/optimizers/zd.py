from typing import Callable, Dict
import numpy as np
from fdopt.direction_generators import DirectionGenerator

from fdopt.optimizers.opt import Optimizer
from fdopt.direction_generators.generator import DirectionGenerator
from fdopt.stepsize.standard_stepsizes import Stepsize

from concurrent.futures import ProcessPoolExecutor as PPE, ThreadPoolExecutor as TPE


class ZD(Optimizer):

    def __init__(self, alpha : Stepsize, h: Stepsize, g_estimator : str = 'forward', **kwargs) -> None:
        assert g_estimator in ['forward', 'single', 'central']
        super().__init__(**kwargs)
        self._approx_grad = getattr(self, '_compute_{}_fd'.format(g_estimator))
        self.alpha = alpha
        self.h = h

    def _compute_forward_fd(self, f : Callable, x : np.ndarray, h : float, P : np.ndarray, executor : PPE | TPE | None = None):
        if executor is None:
            return ((f(x + h * P) - f(x))/h) * P
        g = np.zeros(self.d)
        fx = f(x)
        for i, f_next in enumerate(executor.map(f, [x + h * P[i] for i in range(self.l)])):
            g+= ((f_next - fx)/h) * P[i]
        return g
    
    def optimize(self, f: Callable, x_0: np.ndarray, T: int = 100) -> Dict:
        x = x_0.copy()
            