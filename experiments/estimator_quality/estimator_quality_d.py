import os
import sys
import torch
import time
import pandas as pd
import numpy as np
from math import sqrt
from fdopt.grad_estimators import ForwardFD, CentralFD, OptFFD, OptCFD
from fdopt.direction_generators import Householder, ButterflyMatrix, CoordinateDirections, GaussianDirections, SphericalDirections, QRDirections, SVDDirections


sys.path.append("../")

from synthetic_targets import LeastSquares, SumOfSquares
from utils import relative_error, get_estimator, compute_error_time

methods = {
    'coordinate' : CoordinateDirections,
    'gaussian' : GaussianDirections,
    'spherical' : SphericalDirections,
    'householder' : Householder,
    'butterfly' : ButterflyMatrix,
    'butterfly_dct' : ButterflyMatrix,
    'qr' : QRDirections,
    'svd' : SVDDirections
}




estimator_type = sys.argv[1] # 

out_dir = f"./results/changing_d/{estimator_type}"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)




g = get_estimator(estimator_type, norm_const=1.0) 

fun_name = "LeastSquares"

dtype = torch.float64
device = 'cuda'
seed = 12131419
max_val, min_val = 10.0, -10.0
reps=100
dimensions = [2, 10, 50, 100, 250, 500, 1000] #np.logspace(1, 3, 4).astype(np.int32)
generator = torch.Generator(device=device).manual_seed(12131415)


h = 1e-5

data = []
for (i, d) in enumerate(dimensions):
    
    fun = LeastSquares(mu=1, L=10, d=d, device=device, dtype=dtype)
    x = (max_val - min_val) * torch.rand((1, d), generator = generator, device = device, dtype=torch.float64) + min_val
    fx = fun(x)
    print(f"[++] Computing for d = {d}")
    for (label, matrix_cls) in methods.items():
        P = matrix_cls(d = d, l = d, dct=True, device=device, dtype=dtype) if label == 'butterfly_dct' else  matrix_cls(d = d, l = d, device=device, dtype=dtype)
        g.nrm_const = 1/d if label=='gaussian' else 1.0
        mu_err, std_err, mu_time, std_time = compute_error_time(fun, x.clone(), fx, g, P, h, reps = reps)
        data.append([label, d, mu_err, std_err, mu_time, std_time]) 
        print(f"\t[--] Completed for {label}")

dframe = pd.DataFrame(data, columns=['directions', 'dimension', 'mu_err', 'std_err', 'mu_time', 'std_time'])
dframe.to_csv(f"{out_dir}/{fun_name}.csv", index=False)
