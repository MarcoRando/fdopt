import numpy as np


class DirectionGenerator:
    
    def __init__(self, d : int, l : int = 1, dtype : np.dtype = np.float64, device : str = 'cpu', seed : int = 121212) -> None:
        self.d = d
        self.l = l
        self.dtype = dtype
        self.device = device
        self.seed = seed
        self.rnd_state = np.random.RandomState(seed)
        
    def __call__(self) -> np.ndarray:
        raise NotImplementedError("Implemented in sub-classes!")
    