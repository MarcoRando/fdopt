import os
import sys
import argparse
import torch
import sys
import time
import pandas as pd
import numpy as np
from math import sqrt
from fdopt.direction_generators import * 
from fdopt.optimizers import FDOpt, LSFDOpt

sys.path.append("../")

from nonconvex_functions import StyblinskiTangFunction, RosenbrockFunction, QuingFunction
from utils import compute_error_time, get_grad_estimator, get_target_function, get_grad_cost


methods = {
   'spherical' : SphericalDirections,
   'gaussian' : GaussianDirections,
   'rademacher' : RademacherDirections,
   'coordinate' : CoordinateDirections,
    'householder' : Householder,
    'butterfly' : ButterflyMatrix,
    'qr' : QRDirections,
}


grad_estimator = sys.argv[1]
base_dir = sys.argv[2]

d = int(sys.argv[3])

sigma_noise = float(sys.argv[4])


reps = int(sys.argv[5])

fun_name =sys.argv[6]

out_dir = f"{base_dir}/convergence/noise_{sigma_noise}/{grad_estimator}"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)




dtype = torch.float64
device = 'cuda'
seed = 12131419

if fun_name == 'StyblinskiTang':
    fun = StyblinskiTangFunction(d = d, device=device, dtype = dtype, sigma_noise=sigma_noise)
elif fun_name == 'Rosenbrock':
    fun = RosenbrockFunction(d = d, device=device, dtype = dtype, sigma_noise=sigma_noise)
elif fun_name == 'Quing':
    fun = QuingFunction(d = d, device=device, dtype = dtype, sigma_noise=sigma_noise)
else:
    raise Exception("Not implemented")

g = get_grad_estimator(grad_estimator=grad_estimator, norm_const=1.0)
T = 1000

num_directions = [1, d//3, d//2, round((4/5) * d),  d ]

for h in [1e-7]:
    data = []
    for (i, l) in enumerate(num_directions):
        
        grad_cost = get_grad_cost(grad_estimator, l)
        print(f"[++] Computing for d = {d} l = {l} and h = {h}")
        for (label, matrix_cls) in methods.items():
            generator = torch.Generator(device=device).manual_seed(123123123)
            print(f"[--] Computing for {label}")
            P =  matrix_cls(d = d, l = l, dtype=dtype, device=device)
            g.nrm_const = 1/l if label=='gaussian' or label=='rademacher' else d/l
            gamma = 1e-3 
            opt = LSFDOpt(P = P, g = g, gamma_0=gamma, h = h, grad_cost=grad_cost, gamma_minmax=(1e-16, 1000.0), theta=0.5, rho=2.0, c = 0.001)
            X = (fun.bounds[:, 1] - fun.bounds[:, 0])  * torch.rand((1, d), dtype=dtype, device=device, generator=generator) + fun.bounds[:, 0]
            class_errors = []
            for r in range(reps):
                x0 =  X[0, :].reshape(1, -1).clone()
                f_0 = fun(x0).cpu().item()
                print(f"[--] F0 = {f_0}")
                x, values, num_evals, gammas, grad_norms = opt.optimize(fun, x_0 =x0, T = T)
                with open(f"{out_dir}/{fun_name}_{label}_{l}_{d}_{h}_{r}.csv", 'w') as f:
                    for i in range(len(values)):
                        f.write("{},{},{},{},{},{}\n".format(f_0, fun.min_f, values[i], num_evals[i], gammas[i], grad_norms[i]))
                        f.flush()
