
import torch


class BenchmarkFunction:
    
    def __init__(self, name : str, d : int):
        self.name = name
        self.d = d
        
    def __call__(self, x : torch.Tensor, z = None):
        raise NotImplementedError
    
    def __repr__(self):
        return "{}_{}".format(self.name, self.d)
    
    
class SphereFunction(BenchmarkFunction):
    
    def __init__(self, d : int):
        super().__init__("Sphere", d)
        
    def __call__(self, x, z = None):
        return x.square().sum(dim=1)
    
