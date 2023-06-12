import torch

from fdopt.direction_generators.generator import DirectionGenerator

class GaussianDirections(DirectionGenerator):

    def __call__(self) -> torch.Tensor:
        return torch.randn(size=(self.l, self.d), generator=self.generator,  dtype=self.dtype, device = self.device)

class SphericalDirections(GaussianDirections):
    def __call__(self) -> torch.Tensor:
        P = super().__call__().T
        return (P/P.norm(dim=0)).T

class RademacherDirections(DirectionGenerator):
    def __call__(self) -> torch.Tensor:
        return (torch.rand(size=(self.l, self.d), generator=self.generator,  dtype=self.dtype, device = self.device ) < 0.5).to(dtype=self.dtype).mul_(2).add(-1) 
