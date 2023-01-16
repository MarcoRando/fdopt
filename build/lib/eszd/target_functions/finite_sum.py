import torch

from eszd.target_functions.target_function import TargetFunction

class FiniteSumTarget(TargetFunction):
    
    def __init__(self, name, x_min, f_min, seed, num_elem) -> None:
        super().__init__(name, x_min, f_min, seed)
        self.num_elem = num_elem
        
    def __call__(self, x, *args):
        pass