import torch

class TargetFunction:
    
    def __init__(self, name, x_min, f_min, seed) -> None:
        self.name = name
        self.x_min = x_min
        self.f_min = f_min
        self.seed = seed
        
    def __call__(self, x, *args):
        raise NotImplementedError
