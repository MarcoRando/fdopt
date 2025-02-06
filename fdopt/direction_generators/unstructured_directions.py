import torch
from fdopt.direction_generators.generator import DirectionGenerator

class GaussianDirections(DirectionGenerator):

    def __call__(self) -> torch.Tensor:
        return torch.randn((self.l, self.d), dtype=self.dtype, generator=self.generator, device=self.device)
    
    def __str__(self):
        return f"Gaussian_{self.d}_{self.l}"


class SphericalDirections(GaussianDirections):
    def __call__(self) -> torch.Tensor:
        P = torch.randn(self.d, self.l, dtype=self.dtype, generator=self.generator, device=self.device)
        P.div_(P.norm(dim = 0))
        return P.T
    
    def __str__(self):
        return f"Spherical_{self.d}_{self.l}"

