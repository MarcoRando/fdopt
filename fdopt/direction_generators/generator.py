import torch


class DirectionGenerator:
    
    def __init__(self, d : int, l : int = 1, dtype : torch.dtype = torch.float64, device : str = 'cpu', seed : int = 121212) -> None:
        self.d = d
        self.l = l
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.generator = torch.Generator(device = device)
        self.generator.manual_seed(seed)
        
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError("Implemented in sub-classes!")
    