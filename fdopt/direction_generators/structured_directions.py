import torch
from fdopt.direction_generators.generator import DirectionGenerator




class CoordinateDirections(DirectionGenerator):
    
    def __call__(self) -> torch.Tensor:
        I = torch.zeros((self.l, self.d), dtype=self.dtype, device=self.device)
        inds = torch.randperm(self.d, device=self.device, generator=self.generator)[:self.l]
        I[range(self.l), inds] = 1.0
        return I
    
    def __str__(self):
        return f"COO_{self.d}_{self.l}"

class QRDirections(DirectionGenerator):
    def __call__(self) -> torch.Tensor:
        P = torch.randn(self.d, self.l, dtype=self.dtype, device=self.device, generator=self.generator)
        return torch.linalg.qr(P)[0].T

    def __str__(self):
        return f"QR_{self.d}_{self.l}"

class SingleHouseholder(DirectionGenerator):
    def __init__(self, d: int, l: int = 1, dtype: torch.dtype = torch.float64, device: str = 'cpu', seed: int = 121212) -> None:
        super().__init__(d, l, dtype, device, seed)
        self.I = torch.eye(l, d, dtype=dtype, device=device).to_sparse()

    def __call__(self) -> torch.Tensor:
        v = torch.randn(self.d, dtype=self.dtype, device=self.device, generator=self.generator)
        v /= torch.linalg.norm(v, ord=2)
        return self.I - 2 * v[:self.l].outer(v)

    def __str__(self):
        return f"Householder_{self.d}_{self.l}"
