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
from utils import compute_error_time, get_grad_estimator, get_target_function

methods = {
    'coordinate' : CoordinateDirections,
    'spherical' : SphericalDirections,
    'gaussian' : GaussianDirections,
    'householder' : Householder,
    'butterfly' : ButterflyMatrix,
    'butterfly_dct' : ButterflyMatrix,
    'qr' : QRDirections,
#    'svd' : SVDDirections
}


grad_estimator = sys.argv[1]
base_dir = sys.argv[2]

d = int(sys.argv[3])

sigma_noise = float(sys.argv[4])

L = float(sys.argv[5])
mu = float(sys.argv[6])

reps = int(sys.argv[7])

out_dir = f"{base_dir}/estimator_quality/noise_{sigma_noise}/{grad_estimator}"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)




dtype = torch.float64
device = 'cuda'
seed = 12131419



#h = 5e-1

fun = LeastSquares(mu=mu, L=L, set_l=True, d=d, device=device, dtype=dtype, sigma_noise=sigma_noise)# get_target_function(fun_name, d=d, sigma_noise=sigma_noise, dtype=dtype, device=device) 


d = fun.d # NOTE THAT IT IS DIFFERENT FROM PASSED PARAMETER CUTEST functions!
x = fun.x0
g = get_grad_estimator(grad_estimator=grad_estimator, norm_const=1.0)


num_directions = list(range(1, d, 2)) + [d]

for h in [1.0, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7]:
    data = []
    for (i, l) in enumerate(num_directions):
        
        fx = fun(x)
        print(f"[++] Computing for l = {l} and h = {h}")
        for (label, matrix_cls) in methods.items():
            P = matrix_cls(d = d, l = l, dct=True, dtype=dtype, device=device) if label == 'butterfly_dct' else  matrix_cls(d = d, l = l, dtype=dtype, device=device)
            g.nrm_const = 1/l if label=='gaussian' else d/l
            mu_err, std_err, mu_time, std_time = compute_error_time(fun, x.clone(), g, P, h, reps = reps)
            data.append([label, l, mu_err, std_err, mu_time, std_time]) 
            print(f"\t[--] Completed for {label}")

    dframe = pd.DataFrame(data, columns=['directions', 'l', 'mu_err', 'std_err', 'mu_time', 'std_time'])
    dframe.to_csv(f"{out_dir}/LeastSquare_{L}_{mu}_{d}_{h}.csv", index=False)
