import os
import sys
import argparse
import torch
import sys
import time
import pandas as pd
import numpy as np
from math import sqrt
from fdopt.grad_estimators import ForwardFD, CentralFD
from fdopt.direction_generators import Householder, ButterflyMatrix, CoordinateDirections, GaussianDirections, SphericalDirections, QRDirections, SVDDirections


sys.path.append("../")

from synthetic_targets import LeastSquares
from utils import compute_error_time

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



estimator_type = sys.argv[1]


out_dir = f"./results/changing_l/{estimator_type}"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)




g = ForwardFD(norm_const=1.0)

fun_name = "LeastSquares"

dtype = torch.float64
device = 'cuda'
seed = 12131419
reps=10
#d = 100
d = 50#00


#num_directions = [2, 10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 97, 99, 100]

num_directions = [d, d-1, d-2, d-3, d-4, d - 5, int((5/6) * d ), int((4/5) * d ), int((3/4) * d ), int( (2/3) * d), d//2, d//3, d//10, 2] # [500, 1000, 1500, 2000, 5000, 7000, 9000, 9500, 9600, 9800, 9900, 9990, 10000]
num_directions = num_directions[::-1]
generator = torch.Generator(device=device).manual_seed(12131415)


h = 1e-1#e-3

data = []
x = torch.ones((1, d), dtype=dtype, device=device) #(max_val - min_val) * torch.rand((1, d), generator = generator, device = device, dtype=torch.float64) + min_val
for (i, l) in enumerate(num_directions):
    
    fun = LeastSquares(mu=0.1, L=100, d=d, set_l=True, device=device, dtype=dtype)
    fx = fun(x)
    print(f"[++] Computing for l = {l}")
    for (label, matrix_cls) in methods.items():
        P = matrix_cls(d = d, l = l, dct=True, device=device, dtype=dtype) if label == 'butterfly_dct' else  matrix_cls(d = d, l = l, device=device, dtype=dtype)
        g.nrm_const = 1/l if label=='gaussian' else d/l
        mu_err, std_err, mu_time, std_time = compute_error_time(fun, x.clone(), fx, g, P, h, reps = reps)
        data.append([label, l, mu_err, std_err, mu_time, std_time]) 
        print(f"\t[--] Completed for {label}")

dframe = pd.DataFrame(data, columns=['directions', 'l', 'mu_err', 'std_err', 'mu_time', 'std_time'])
dframe.to_csv(f"{out_dir}/{fun_name}_{d}.csv", index=False)
