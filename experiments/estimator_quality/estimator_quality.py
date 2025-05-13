import os
import sys
import argparse as ap
import torch
import sys
import time
import pandas as pd
import numpy as np
from math import sqrt
from fdopt.direction_generators import *
from fdopt.grad_estimators import ForwardFD

sys.path.append("../")

from synthetic_targets import LeastSquares, RegularizedLogistic, SparseSum
from utils import compute_error_time, get_grad_estimator, get_target_function

methods = {
    'coordinate' : CoordinateDirections,
    'spherical' : SphericalDirections,
    'rademacher' : RademacherDirections,
    'gaussian' : GaussianDirections,
    'householder' : Householder,
    'permuted_householder' : PermutedHouseholder,
    'butterfly' : ButterflyMatrix,
    'qr' : QRDirections,
}



def run_experiment(args):
    fun_name = args.fun_name
    base_dir = args.out_dir
    dtype = torch.float64
    device = 'cuda'
    seed = 12131419

    d = args.d
    num_directions = [2, d//3, d//2, round((4/5) * d),  d] 
    sigma_noise = args.sigma

    L, mu = args.L, args.mu

    h = args.h
    reps = args.reps 
    num_init = args.num_init
    n = args.n
    s = args.s
    lam = args.lam
    g = ForwardFD(norm_const=1.0) #get_grad_estimator(grad_estimator=grad_estimator, norm_const=1.0)


    if fun_name == 'least_squares':
        fun = LeastSquares(d, mu=mu, L=L, set_l=True, dtype=dtype, device=device, sigma_noise=sigma_noise, seed=seed)
    elif fun_name == 'logistic':
        fun = RegularizedLogistic(d, n, lam=lam, dtype=dtype, device=device, sigma_noise=sigma_noise, seed=seed)
    elif fun_name == 'sparse_sum':
        fun = SparseSum(d, s=s, dtype=dtype, device=device, sigma_noise=sigma_noise, seed=seed)
    else:
        fun =get_target_function(fun_name)(d, dtype=dtype, device=device, sigma_noise=sigma_noise, seed=seed)

    out_dir = f"{base_dir}/grad_approx_error/noise_{sigma_noise}/ffd"

    os.makedirs(out_dir, exist_ok=True)

    data = []
    for (i, l) in enumerate(num_directions):        
        print(f"[++] Computing for d = {d} l = {l} and h = {h}")
        for (label, matrix_cls) in methods.items():
            generator = torch.Generator(device=device).manual_seed(123123123)
            P =  matrix_cls(d = d, l = l, dtype=dtype, device=device)
            if label == 'gaussian' or label =='rademacher':
                g.nrm_const = 1/l
            else:
                g.nrm_const = d/l
            X = (fun.bounds[:, 1] - fun.bounds[:, 0]) * torch.rand((num_init, d), dtype=dtype, device=device, generator=generator) + fun.bounds[:, 0]
            for point_idx in range(X.shape[0]):
                errors, times = compute_error_time(fun, X[point_idx, :].reshape(1, -1), g, P, h, reps = reps)
                for j in range(len(errors)):
                    data.append([label, l, errors[j], times[j], j, point_idx]) 
            print(f"\t[--] Completed for {label}")

    dframe = pd.DataFrame(data, columns=['directions', 'l', 'err', 'time', 'rep', 'point_idx'])
    dframe.to_csv(f"{out_dir}/{str(fun)}_{h}.csv", index=False)





if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Compute gradient approximation error for a given target.", formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('fun_name', type=str)
#    parser.add_argument('grad_estimator', type=str, default='ffd')
    parser.add_argument('d', type=int, default=50)

    # Parameters for Least Squares
    parser.add_argument('--L', type=float, default=10000.0)
    parser.add_argument('--mu', type=float, default=1.0)

    # Parameters for Logistic
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--lam', type=float, default=1e-5)

    # Sparse Sum
    parser.add_argument('--s', type=int, default=10)


    parser.add_argument('--out_dir', type=str, default='./')
    parser.add_argument('--reps', type=int, default=50)
    parser.add_argument('--num_init', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=0.0)

    parser.add_argument('--h', type=float, default=1e-7)
    args = parser.parse_args()
    run_experiment(args)