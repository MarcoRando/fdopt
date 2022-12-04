from eszd.direction_generator import DirectionMatrix

class Optimizer:
    def __init__(self, P : DirectionMatrix, alpha, h):
        self.P = P        
        self.alpha = alpha
        self.h = h
        
    def step(self, fun, x, *args):
        pass
    
    def reset(self):
        pass
        