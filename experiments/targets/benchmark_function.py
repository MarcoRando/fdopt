import torch
import numpy as np
from typing import Any

class BenchmarkFunction:
    
    def __init__(self, name : str, d : int, x_star : torch.Tensor | None = None, fx_star : float | None = None, sigma : float = 0.0, seed : int = 121212, device : str = 'cpu' , dtype : torch.dtype = torch.float64) -> None:
        assert x_star is None or x_star.shape[1] == d
        self.name = name
        self.fx_star = fx_star
        self.d = d
        self.sigma = sigma
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.generator = torch.Generator(device = device)
        self.generator.manual_seed(seed)
                
    def add_noise(self, fx : torch.Tensor):
        eps = torch.randn(fx.shape, device=self.device, dtype=self.dtype, generator=self.generator)
        return fx + self.sigma * eps
        
        
    def __call__(self, x : torch.Tensor) -> float:
        pass

    def __str__(self) -> str:
        return "{} [d = {}]".format(self.name, self.d)
    
    def __repr__(self) -> str:
        return "{}_{}".format(self.name, self.d)