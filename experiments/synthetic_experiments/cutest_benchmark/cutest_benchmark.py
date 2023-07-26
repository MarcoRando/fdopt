import sys

import time
import os
import torch
import numpy as np
from math import sqrt
import pycutest as pct


from fdopt import ZD, PFZD
from fdopt.grad_estimators import CentralGenStructGradApprox, CentralGenGradApprox, GenGradApprox, GenStructGradApprox, SinglePointEstimator, ForwardFD, CentralFD, LinInterpolation, CentralLinInterpolation
from fdopt.stepsize import PolyakStepsize, ConstantStepsize, Stepsize
from fdopt.direction_generators import GaussianDirections, RademacherDirections, SphericalDirections, CoordinateDirections, QRDirections

from target_functions import TestQuad, Dixon3DQ, TRIDIA
sys.path.append("../../")
from utils import ConvCriterion

dtype = torch.float64
device = 'cpu'

f_L = 0.0
taus = [1e-3, 1e-4, 1e-5]#, 1e-8]

budget = 10000 #num_iters * l * num_split
reps = 100
d = 30
l = 20
#P = CoordinateDirections(d = d, l = l)
P = CoordinateDirections(d = d, l =l)
fun = TRIDIA(d = d)
g = ForwardFD(d = d, l =l, f = fun, P=P)
for _ in range(1):
    opt = ZD(g=g, gamma=PolyakStepsize(gamma_0=1.0, f_min=fun.fx_star), h=ConstantStepsize(1e-3))
    #pfzd = PFZD(opt)
    x0 = torch.Tensor(fun.x0).reshape(1, d)
    ris = opt.optimize(x_0=x0, T = budget, convergence_criterion=ConvCriterion(1e-5, f_L = fun.fx_star))
    print(ris['converged'], ris['iterations'] * l)
    print(ris['fx'].item(), fun.fx_star)
    opt.reset()