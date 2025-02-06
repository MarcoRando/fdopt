import torch
from fdopt import FDOpt
from fdopt.direction_generators import QRDirections
from fdopt.grad_estimators import ForwardFD


def target(x):
    
    return x.square().sum(dim=1, keepdims=True) 


d = 5
l = 2
dtype = torch.float32
device = 'cuda'
seed = 121314

gamma = 0.01
h = 1e-3

x = torch.ones((1, d), dtype=dtype, device=device)
P = QRDirections(d = d, l = l, dtype = dtype, device = device, seed = seed)
g = ForwardFD(norm_const = d / l)


#print(x, target(x))

opt = FDOpt(P = P, g = g, gamma = gamma, h = h, seed = seed)

opt.optimize(target, x, T = 100)