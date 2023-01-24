import torch
from eszd.directions.directions import DirectionMatrix

   
        
class GaussianDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        return torch.randn(self.d, self.l, generator=self.generator, device=self.device, dtype=self.dtype)
       
       
class SphericalDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        A /= torch.linalg.norm(A, 2, dim=0)
        return  A
    
class RadamacherDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        return  2 * (torch.rand(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype) > 0.5) - 1
        