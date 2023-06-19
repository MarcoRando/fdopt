import torch
import sys
import os
import numpy as np
from fdopt import ZD
from fdopt.grad_estimators import GenGradApprox, GenStructGradApprox, SinglePointEstimator, ForwardFD, CentralFD, LinInterpolation, CentralLinInterpolation
from fdopt.direction_generators import GaussianDirections, SphericalDirections, CoordinateDirections, QRDirections

from utils import compute_relative_error, get_g, get_methods, get_gsga_methods, compute_grad_accuracy, store_result

sys.path.append("../../targets/")
from synthetic import LeastSquares,SphereFunction

device = 'cpu'
dtype = torch.float64


out_path = "../results/grad_accuracy/changing_l/"
os.makedirs(out_path, exist_ok=True)            

L = 20
mu = 5
d = 100
dimensions = [1] + [i for i in range(5, 105,5)]#, 2, 3, 4, 5]#, 10, 20, 50, 100, 250, 450, 500]#, 10, 15, 20, 25, 30, 40, 50]#[i for i in range(2, 11)] #+ [15, 17, 20, 25, 30, 40, 50, 100, 200, 500]
reps = 200
h = 1e-10
with open(f"{out_path}/exp_info.log", 'w') as f:
    f.write(f"[--] L = {L}\n")
    f.write(f"[--] mu = {mu}\n")
    f.write(f"[--] dimensions = {repr(dimensions)}\n")
    f.write(f"[--] device = {device}\n")
    f.write(f"[--] dtype = {dtype}\n")
    f.write(f"[--] reps = {reps}\n")
    f.write(f"[--] h = {h}\n")

t1 = SphereFunction(d = d, device=device, dtype=dtype) #LeastSquares(m = d, d=d, device=device, dtype=dtype, L = L, mu =mu)

x1 = torch.ones((1, d), device=device, dtype=dtype)
x2 = torch.tensor([i for i in range(d)], dtype=dtype, device=device).reshape(1, d) + 1e-10
x3 = torch.tensor([i**2 + d for i in range(d)], device=device, dtype=dtype).reshape(1, d)


for l in dimensions:
    print(f"[++] Executing experiment with d = {d} l = {l}")

    methods = get_methods()
    methods = get_gsga_methods(methods)
    objs = [(t1, [x1, x2, x3])]

    results = compute_grad_accuracy(objs, methods,  d= d, l = l, h = h, reps=reps, device=device, dtype=dtype)

    store_result(results, l, out_path)

