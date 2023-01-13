import torch
from eszd.directions.directions import DirectionMatrix

class StructuredSphericalDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        return self.nrm_const * torch.linalg.qr(A, mode='reduced')[0]
    