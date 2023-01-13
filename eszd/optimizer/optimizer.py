class Optimizer:
    def __init__(self, target, x0) -> None:
        self.target = target
        self.x = x0
        self.t = 1
        
    def step(self):
        raise NotImplementedError
    