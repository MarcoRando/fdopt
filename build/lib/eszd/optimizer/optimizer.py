class Optimizer:
        
    def __init__(self, name : str, verbose : bool = False) -> None:
        self.name = name
        self.verbose = verbose
        
    def optimize(self, target, x, T = 1000, eps = 1e-5):
        raise NotImplementedError
    