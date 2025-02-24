import os
import sys
import argparse
import torch
import sys
import time
import pandas as pd
import numpy as np
from math import sqrt
import pycutest as pc 
from fdopt.grad_estimators import ForwardFD, CentralFD
from fdopt.direction_generators import Householder, ButterflyMatrix, CoordinateDirections, GaussianDirections, SphericalDirections, QRDirections, SVDDirections


sys.path.append("../")

from synthetic_targets import LeastSquares, PycutestTarget
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

reps = int(sys.argv[5])

seed = 131415

out_dir = f"{base_dir}/estimator_quality/noise_{sigma_noise}/{grad_estimator}"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)


def get_pycutest_target(problem_name):
    name, d, _ = problem_name.split(',')
    
    d = int(d)

    print("[--] problem = {}\td = {}".format(name, d))
    problem = pc.import_problem(name, sifParams={'N' : d})
    return PycutestTarget(problem, dtype=dtype, device=device, seed = seed)
    
dtype = torch.float64
device = 'cuda'
seed = 12131419

problems = [
    'ALJAZZAF,1000,3',
]

#h = 5e-1


for problem in problems:
    fun = get_pycutest_target(problem)
    
    X = torch.ones((1, fun.d))
     
    eval_time = time.time()
    print("[--] f(x0) = {}".format(fun(X)))
    eval_time = time.time() - eval_time
    print("[--] grad = {}".format(fun.grad(X[0])))
    print(eval_time)
    
# fun = PycutestTarget(d =)


# d = fun.d # NOTE THAT IT IS DIFFERENT FROM PASSED PARAMETER CUTEST functions!
# x = fun.x0
# g = get_grad_estimator(grad_estimator=grad_estimator, norm_const=1.0)


# num_directions = list(range(1, d, 2)) + [d]

# for h in [1.0, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7]:
#     data = []
#     for (i, l) in enumerate(num_directions):
        
#         fx = fun(x)
#         print(f"[++] Computing for l = {l} and h = {h}")
#         for (label, matrix_cls) in methods.items():
#             P = matrix_cls(d = d, l = l, dct=True, dtype=dtype, device=device) if label == 'butterfly_dct' else  matrix_cls(d = d, l = l, dtype=dtype, device=device)
#             g.nrm_const = 1/l if label=='gaussian' else d/l
#             mu_err, std_err, mu_time, std_time = compute_error_time(fun, x.clone(), g, P, h, reps = reps)
#             data.append([label, l, mu_err, std_err, mu_time, std_time]) 
#             print(f"\t[--] Completed for {label}")

#     dframe = pd.DataFrame(data, columns=['directions', 'l', 'mu_err', 'std_err', 'mu_time', 'std_time'])
#     dframe.to_csv(f"{out_dir}/LeastSquare_{L}_{mu}_{d}_{h}.csv", index=False)
