import torch
import sys
import numpy as np 
import pandas as pd
import os
import time
sys.path.append("../")

from fdopt.optimizers import FDOpt
from fdopt.grad_estimators import ForwardFD
from fdopt.direction_generators import *
from fdopt.stepsize import PolyakStep
from synthetic_targets import LeastSquares


methods = {
   'householder' : Householder,
   'coordinate' : CoordinateDirections,
   'gaussian' : GaussianDirections,
   'spherical' : SphericalDirections,
   'householder' : Householder,
   'butterfly' : ButterflyMatrix,
   'butterfly_dct' : ButterflyMatrix,
   'qr' : QRDirections,
#    'svd' : SVDDirections
}


d = 100
l = d//2
device = 'cuda'
dtype=torch.float64



fun_name = "LeastSquares"

seed = 12131419
reps=5
generator = torch.Generator(device=device).manual_seed(12131415)


h = lambda t : float(sys.argv[1]) / (t + 1)

data = []
target = LeastSquares(mu=0.01, L=10.0, set_l=True, d=d, dtype=dtype,device=device)
x = torch.ones((1, d), dtype=dtype, device=device) #(max_val - min_val) * torch.rand((1, d), generator = generator, device = device, dtype=torch.float64) + min_val
out_dir = "./results/conv_speed/ffd"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)

mean_values = {}

f_0 = target(x).item()
print("Computing for Polyak stepsize")
for (label, matrix_cls) in methods.items():
    

    norm_const = d/l if label != 'gaussian' else 1/l
    g = ForwardFD(norm_const=norm_const)

    gamma = PolyakStep(gamma_0=1.0, min_f=0.0)
    if label not in mean_values:
        mean_values[label] = (None, [], [])
    P = matrix_cls(d = d, l = l, dct=True, device=device, dtype=dtype) if label == 'butterfly_dct' else  matrix_cls(d = d, l = l, device=device, dtype=dtype)
    opt = FDOpt(P = P, g =g, gamma=gamma, h=h)
    last_vals = []
    opt_times = []
    values = []
    for _ in range(reps):
        it_time = time.time()
        _, f_values = opt.optimize(f = target, x_0 = x.clone(), T = 1000)
        it_time = time.time() - it_time
        opt_times.append(it_time)
        values += f_values
    values = np.array(values).reshape(reps, -1)
    mean_values[label] = (np.mean(values,axis=0), np.std(values,axis=0))



for label in methods.keys():
    with open(f"{out_dir}/{fun_name}_{d}_{l}_{h(0)}_{label}_best", 'w') as f:
        mu_vals, std_vals = mean_values[label]
        for i in range(len(mu_vals)):
            f.write("{},{}\n".format(mu_vals[i], std_vals[i]))

