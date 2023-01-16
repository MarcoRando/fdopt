import torch
import os
from benchmark_functions import SphereFunction

from eszd import SSZD
from eszd.target_functions import FiniteSumTarget
from eszd.directions import SphericalDirections, StructuredSphericalDirections, GaussianDirections
import numpy as np

from utils import *

class LinearRegression(FiniteSumTarget):
    #    def __init__(self, name, x_min, f_min, seed, num_elem) -> None:
    def __init__(self, d, num_elems, seed, device = 'cpu', dtype=torch.float32) -> None:
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.w = torch.rand(size=(1, d), generator = self.generator, device=device, dtype=dtype)
        self.X = torch.rand((num_elems, d), generator=self.generator, device=device, dtype=dtype) * 10 + 5
        self.y = torch.matmul(self.w,self.X.T).reshape(-1)
        super().__init__("Linear Regression", self.w, 0.0, seed, num_elems)

    def __call__(self, w_hat, idx = None):
        if idx is None:
            return (torch.matmul(w_hat , self.X.T) - self.y).square().mean(axis=1)
#        print((torch.matmul(w_hat , self.X[idx, :]) - self.y[idx]).square())
        return (torch.matmul(w_hat , self.X[idx, :]) - self.y[idx]).square()


os.makedirs("./results/comparisons", exist_ok=True)

d = 1000# int(1e4)
l = 1000 #// 2

dtype = torch.float64
target = LinearRegression(d=d, num_elems=100, seed=1212, device='cpu', dtype=dtype)#SphereFunction(d)
#    def __init__(self, target, x0, P : DirectionMatrix, alpha, h):
L = torch.linalg.eigvalsh(target.X.T @ target.X).max()
x = torch.zeros(1, d, dtype=dtype)
print(target(target.w))#, target.w)
alpha = lambda t : l/(d * L * sqrt(t))
h = lambda t : 1e-7

sampler = torch.Generator()
sampler.manual_seed(121314)


def sampling_procedure():
    return torch.randint(low=0, high=target.X.shape[0], size=(1,), generator=sampler).item()

opt = SSZD(ButterflyDirections(d=d, l=l, dtype=dtype), alpha, h, sampling_procedure=sampling_procedure)
#
##    def optimize(self, target, x, T = 1000, eps = 1e-5):
#T = 20
x, mean_g_time =opt.optimize(target, x, T=int(1e6), eps=1e-5)
print(np.mean(mean_g_time))
#print(x)


