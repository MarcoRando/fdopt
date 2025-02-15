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
from synthetic_targets import LeastSquares


methods = {
#    'coordinate' : CoordinateDirections,
#    'gaussian' : GaussianDirections,
    'spherical' : SphericalDirections,
    'householder' : Householder,
#    'butterfly' : ButterflyMatrix,
#    'butterfly_dct' : ButterflyMatrix,
#    'qr' : QRDirections,
#    'svd' : SVDDirections
}


d = 100
l = d
device = 'cuda'
dtype=torch.float64


target = LeastSquares(mu=0.01, L=10.0, set_l=True, d=d, dtype=dtype,device=device)

g = ForwardFD(norm_const=d/l)

fun_name = "LeastSquares"

dtype = torch.float64
device = 'cuda'
seed = 12131419
reps=5
generator = torch.Generator(device=device).manual_seed(12131415)


h = 1e-5

data = []
x = torch.ones((1, d), dtype=dtype, device=device) #(max_val - min_val) * torch.rand((1, d), generator = generator, device = device, dtype=torch.float64) + min_val
out_dir = "./results/tolerance/ffd"#.format(mu, L)
out_dir2 = "./results/conv_speed/ffd"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)
os.makedirs(out_dir2, exist_ok=True)

best_values = {'spherical' : (None, [], []), 'householder' : (None, [], [])}

gammas = np.logspace(-3, 0, 20)
for (i, gamma) in enumerate(gammas):

    f_0 = target(x).item()
    print(f"[{i}] Computing for gamma = {gamma}")
    for (label, matrix_cls) in methods.items():
        P = matrix_cls(d = d, l = l, dct=True, device=device, dtype=dtype) if label == 'butterfly_dct' else  matrix_cls(d = d, l = l, device=device, dtype=dtype)
        opt = FDOpt(P = P, g =g, gamma=gamma, h=h, seed=seed)
        last_vals = []
        opt_times = []
        values = []
        for _ in range(reps):
            it_time = time.time()
            _, f_values = opt.optimize(f = target, x_0 = x, T = 1000)
            it_time = time.time() - it_time
            if f_values[-1] > f_0:
                last_vals.append(f_0)
            else:
                last_vals.append(f_values[-1])
            opt_times.append(it_time)
            values += f_values
        mu_lv, std_lv = np.mean(last_vals), np.std(last_vals)
        mu_time, std_time = np.mean(opt_times), np.std(opt_times)
        data.append([label, gamma, mu_lv, std_lv, mu_time, std_time]) 
        if best_values[label][0] is None or best_values[label][0] > mu_lv:
#            print(mu_lv)
            values = np.array(values).reshape(reps, -1)
            best_values[label] = (mu_lv, np.mean(values,axis=0), np.std(values,axis=0))


dframe = pd.DataFrame(data, columns=['directions', 'gamma', 'mu_last_val', 'std_last_val', 'mu_time', 'std_time'])
dframe.to_csv(f"{out_dir}/{fun_name}_{d}.csv", index=False)

for label in methods.keys():
    with open(f"{out_dir2}/{fun_name}_{d}_{label}_best", 'w') as f:
        _, mu_vals, std_vals = best_values[label]
        for i in range(len(mu_vals)):
            f.write("{},{}\n".format(mu_vals[i], std_vals[i]))

