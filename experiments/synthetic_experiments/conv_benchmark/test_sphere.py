import sys

sys.path.append("../../targets/")
import time
import torch
import numpy as np
from math import sqrt


from fdopt import ZD, PFZD
from fdopt.grad_estimators import CentralGenStructGradApprox, CentralGenGradApprox, GenGradApprox, GenStructGradApprox, SinglePointEstimator, ForwardFD, CentralFD, LinInterpolation, CentralLinInterpolation
from fdopt.stepsize import ConstantStepsize, Stepsize
from fdopt.direction_generators import GaussianDirections, RademacherDirections, SphericalDirections, CoordinateDirections, QRDirections


from synthetic import SphereFunction

d = 50
l = 5
dtype = torch.float64
device = 'cpu'

target = SphereFunction(d = d, dtype = dtype, device=device)

gamma = ConstantStepsize(gamma_0=0.15)
P = QRDirections(d = d, l = d, dtype=dtype, device=device)
g = ForwardFD(d =d, l = l, f = target, P = P)
zd = ZD(g = g, gamma=gamma, h = ConstantStepsize(gamma_0=1e-7))
pfzd = PFZD(zd)
num_iters = 10
num_split = 5
budget = num_iters * l * num_split
print("num f evals", budget)
x0 = torch.tensor([i for i in range(d)]).reshape(1, d)
T = 1000
ris = pfzd.optimize(x_0=x0, T = budget, gamma_space=(1e-10,1e-1), num_split=num_split,return_trace=True)
print(ris['fx'])

