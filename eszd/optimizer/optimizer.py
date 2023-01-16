class Optimizer:
        
    def __init__(self, name) -> None:
        self.name = name
        
    def step(self):
        raise NotImplementedError
    