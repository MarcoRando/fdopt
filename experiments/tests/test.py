import torch

from fdopt.direction_generators import Householder


d = 200
l = 1
dtype=torch.float64
device = 'cuda'


Pg = Householder(d = d, l = l, dtype=dtype,device=device)


P = Pg()

print(P.T @ P)

