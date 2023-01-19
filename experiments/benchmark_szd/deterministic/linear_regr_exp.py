import torch
from eszd import SSZD
from eszd.directions import RandomCoords


def target(x, z=None):
    return x.square().sum(dim=1, keepdims=True)


P = RandomCoords(d=5, l=3)

opt = SSZD(P, alpha=lambda t: 0.5, h = lambda t: 1e-5, verbose=True)

x = torch.ones((5, 1))

ris = opt.optimize(target, x, T=100)
