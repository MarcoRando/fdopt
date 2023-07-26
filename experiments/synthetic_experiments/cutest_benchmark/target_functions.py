

import sys
import pycutest as pct
import torch

sys.path.append("../../targets")
from benchmark_function import BenchmarkFunction

class PyCutestBF(BenchmarkFunction):
    def __init__(self, name : str, d: int, d_defined, sifParams,**kwargs) -> None:
        assert d in d_defined, f"d must be in {d_defined}"
        super().__init__(name = name, d=d, **kwargs)
        self.fun = pct.import_problem(name, sifParams=sifParams)
    
    @property
    def x0(self):
        return self.fun.x0
        
    def __call__(self, x: torch.Tensor) -> float:
        y = torch.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            y[i] = self.fun.obj(x[i, :])
        return self.add_noise(y)
    


class TestQuad(PyCutestBF):
    
    def __init__(self, d: int, dtype = torch.float64, device='cpu', **kwargs) -> None:
        d_defined = [1000, 5000]
        x_star = torch.zeros((1, d), dtype=dtype, device=device)
        fx_star = 0.0
        super().__init__(name="TESTQUAD", d=d, d_defined=d_defined, sifParams=dict(N = d), x_star = x_star, fx_star =0.0, dtype=dtype, device=device, **kwargs)    
        
    
class Dixon3DQ(PyCutestBF):
    
    def __init__(self, d: int, dtype = torch.float64, device='cpu', **kwargs ) -> None:
        x_star = torch.ones((1, d), dtype = dtype, device=device)
        d_defined = [10, 100, 1000, 10000]
        super().__init__(name="DIXON3DQ", d=d, d_defined=d_defined, sifParams=dict(N = d), x_star = x_star, fx_star =0.0, dtype=dtype, device=device, **kwargs)    
    
class TRIDIA(PyCutestBF):
    def __init__(self, d: int, dtype = torch.float64, device='cpu', **kwargs ) -> None:
        x_star = torch.zeros((1, d), dtype = dtype, device=device)
        d_defined = [10, 20, 30, 50, 100, 500, 1000, 5000, 10000]
        super().__init__(name="TRIDIA", d=d, d_defined=d_defined, sifParams=dict(N = d), x_star = x_star, fx_star =1.0, dtype=dtype, device=device, **kwargs)    
    