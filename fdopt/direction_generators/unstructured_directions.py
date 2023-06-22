import torch

from fdopt.direction_generators.generator import DirectionGenerator

class GaussianDirections(DirectionGenerator):

    def __call__(self) -> torch.Tensor:
        return torch.randn(self.l, self.d, generator=self.generator, device=self.device, dtype=self.dtype)
    def __str__(self):
        return f"G_{self.d}_{self.l}"

class SphericalDirections(GaussianDirections):
    def __call__(self) -> torch.Tensor:
        P = torch.randn(self.d, self.l, generator=self.generator, device=self.device, dtype=self.dtype)
        P.div_(torch.linalg.norm(P, dim=0, ord=2))
        return P.T
    
    def __str__(self):
        return f"SP_{self.d}_{self.l}"


class RademacherDirections(DirectionGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob_vec = torch.tensor([1.0, 1.0], device=self.device)

    def __call__(self) -> torch.Tensor:
        return (2 * torch.multinomial(self.prob_vec, self.d * self.l, replacement=True, generator=self.generator) - 1).reshape(self.l, self.d).to(dtype=self.dtype)

    def __str__(self):
        return f"RAD_{self.d}_{self.l}"
