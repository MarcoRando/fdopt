import torch
from eszd.directions.directions import DirectionMatrix

   
        
class GaussianDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        return torch.randn(self.l, self.d, generator=self.generator, device=self.device, dtype=self.dtype)
       
       
class SphericalDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        A /= torch.linalg.norm(A, 2, dim=0)
        return  A.T
    
class RadamacherDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        P = torch.rand(size=(self.l, self.d), generator=self.generator, device=self.device, dtype=self.dtype) > 0.5
        return  (2.0 * P - 1.0).to(dtype=self.dtype)
        
class OnesDirections(DirectionMatrix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.d = torch.ones((self.l, self.d), generator = self.generator, device = self.device, dtype = self.dtype)
    
    def __call__(self):
        return self.d
