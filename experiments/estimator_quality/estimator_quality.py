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
    'gaussian' : GaussianDirections,
    'spherical' : SphericalDirections,
    'householder' : Householder,
    'butterfly' : ButterflyMatrix,
    'butterfly_dct' : ButterflyMatrix,
    'qr' : QRDirections,
#    'svd' : SVDDirections
}


fun_name = sys.argv[1]
grad_estimator = sys.argv[2]
base_dir = sys.argv[3]

d = int(sys.argv[4])
h = float(sys.argv[5])

reps = int(sys.argv[6])

out_dir = f"{base_dir}/estimator_quality/{grad_estimator}"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)




dtype = torch.float64
device = 'cuda'
seed = 12131419



#h = 5e-1

data = []
fun = get_target_function(fun_name, d=d, device=device, dtype=dtype) #LeastSquares(mu=0.1, L=1000, d=d, set_l=False, device=device, dtype=dtype)

d = fun.d # NOTE THAT IT IS DIFFERENT FROM PASSED PARAMETER CUTEST functions!
x = torch.zeros_like(fun.x0)
g = get_grad_estimator(grad_estimator=grad_estimator, norm_const=1.0)


num_directions = [2] #list(range(1, d, 2)) + [d]

generator = torch.Generator(device=device).manual_seed(12131415)


for (i, l) in enumerate(num_directions):
    
    fx = fun(x)
    print(f"[++] Computing for l = {l} and h = {h}")
    for (label, matrix_cls) in methods.items():
        P = matrix_cls(d = d, l = l, dct=True, device=device, dtype=dtype) if label == 'butterfly_dct' else  matrix_cls(d = d, l = l, device=device, dtype=dtype)
        g.nrm_const = 1/l if label=='gaussian' else d/l
        mu_err, std_err, mu_time, std_time = compute_error_time(fun, x.clone(), fx, g, P, h, reps = reps)
        data.append([label, l, mu_err, std_err, mu_time, std_time]) 
        print(f"\t[--] Completed for {label}")

dframe = pd.DataFrame(data, columns=['directions', 'l', 'mu_err', 'std_err', 'mu_time', 'std_time'])
dframe.to_csv(f"{out_dir}/{fun_name}_{fun.d}_{h}.csv", index=False)
