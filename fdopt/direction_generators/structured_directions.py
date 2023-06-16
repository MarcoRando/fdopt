import torch
from fdopt.direction_generators.generator import DirectionGenerator




class CoordinateDirections(DirectionGenerator):
    
    def __call__(self) -> torch.Tensor:
        I = torch.zeros((self.l, self.d), dtype=self.dtype, device=self.device)
        inds = torch.randperm(self.d, device=self.device, generator=self.generator)[:self.l]
        I[range(self.l), inds] = 1.0
        return I

class QRDirections(DirectionGenerator):
    def __call__(self) -> torch.Tensor:
        P = torch.randn(self.d, self.l, dtype=self.dtype, device=self.device, generator=self.generator)
        return torch.linalg.qr(P)[0].T