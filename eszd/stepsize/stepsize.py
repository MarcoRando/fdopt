import numpy as np

class Stepsize:
    def __init__(self, init_value, mode):
        self.init_value = init_value
        self.mode = mode
        self.t = 1
        self.get_value = getattr(self, "_{}".format(mode))
        
    def constant(self):
        return self.init_value
    
    def sqrt(self):
        return self.init_value/np.sqrt(self.t)

    def linear(self):
        return self.init_value/self.t
    
    def __call__(self):
        v = self.get_value()
        self.t+=1
        return v