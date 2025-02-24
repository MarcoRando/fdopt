

class Stepsize:
    
    def __call__(self, k, fx, g_k):
        pass    


class PolyakStep(Stepsize):
    
    def __init__(self, gamma_0, min_f = None, eps_k = lambda t : 0.0, approx_min = False):
        self.gamma_0 = gamma_0
        self.min_f = min_f
        self.eps_k = eps_k
        self.approx_min = approx_min
        
    def __call__(self, k, fx, g_k):
        if self.approx_min and (self.min_f is None or fx < self.min_f):
            self.min_f = fx
            
        return self.gamma_0 * ((fx - self.min_f + self.eps_k(k)) / g_k.flatten().norm(p=2).square())
    
    
class LineSearchStep(Stepsize):
    
    def __init__(self, c = 0.1, ):
        super().__init__()
        
        
    def __call__(self, k, fx, g_k):
        return super().__call__(k, fx, g_k)