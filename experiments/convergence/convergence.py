import os
import sys
import argparse as ap
import torch
import sys
import time
import pandas as pd
import numpy as np
from math import sqrt
from fdopt.direction_generators import * #Householder, ButterflyMatrix, CoordinateDirections, GaussianDirections, SphericalDirections, QRDirections
from fdopt.optimizers import FDOpt, LSFDOpt
from fdopt.grad_estimators import ForwardFD

sys.path.append("../")

from synthetic_targets import LeastSquares, RegularizedLogistic
from utils import compute_error_time, get_target_function

methods = {
    'spherical' : SphericalDirections,
    'gaussian' : GaussianDirections,
    'rademacher' : RademacherDirections,
    'coordinate' : CoordinateDirections,
    'householder' : Householder,
    'butterfly' : ButterflyMatrix,
    "permuted_householder" : PermutedHouseholder,
    'qr' : QRDirections,
}




dtype = torch.float64
device = 'cuda'
seed = 12131419

def run_experiment(args):


    fun_name = args.fun_name

    base_dir = args.out_dir

    d = args.d
    num_directions = [d, round((4/5) * d), d//2, d //3, 2] #[ 2,  d//3, d//2, round((4/5) * d),  d] 

    L, mu = args.L, args.mu
    T = args.T

    h = args.h
    reps = args.reps 
    n = args.n
    lam = args.lam

    g = ForwardFD(norm_const = 1.0) # get_grad_estimator(grad_estimator=grad_estimator, norm_const=1.0)


    if fun_name == 'least_squares':
        fun = LeastSquares(d, mu=mu, L=L, set_l=True, dtype=dtype, device=device,  seed=seed)
    elif fun_name == 'logistic':
        fun = RegularizedLogistic(d, n, lam=lam, dtype=dtype, device=device, seed=seed)
    else:
        fun =get_target_function(fun_name)(d, dtype=dtype, device=device, seed=seed)
    out_dir = f"{base_dir}/convergence/ffd"#{grad_estimator}"#.format(mu, L)

    os.makedirs(out_dir, exist_ok=True)

    data = []
    for (i, l) in enumerate(num_directions):
        
        print(f"[++] Computing for l = {l} and h = {h}")
        for (label, matrix_cls) in methods.items():
            generator = torch.Generator(device=device).manual_seed(123123123)
            print(f"[--] Computing for {label}")
            P = matrix_cls(d = d, l = l, dtype=dtype, device=device)
            g.nrm_const = 1/l if label=='gaussian' or label=='rademacher' else d/l
            gamma = 1.0 
            #c = 0.00001
            opt = LSFDOpt(P = P, g = g, gamma_0=gamma, h = h,  gamma_minmax=(1e-10, 1000.0), theta=0.5, rho=2.0, c = 1e-7)
            x0 = fun.x0
            f_0 = fun(x0).cpu().item()
            for r in range(reps):
                _, values, num_evals, gammas, it_times = opt.optimize(fun, x_0 = x0, T = T)
                with open(f"{out_dir}/{str(fun)}_{label}_{l}_{h}_{r}.csv", 'w') as f:
                    for i in range(len(values)):
                        f.write("{},{},{},{},{},{}\n".format(f_0, fun.min_f, values[i], num_evals[i], gammas[i], it_times[i]))
                        f.flush()
#            print(values[-1] / f_0)


if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Compute gradient approximation error for a given target.", formatter_class=ap.ArgumentDefaultsHelpFormatter)

    parser.add_argument('fun_name', type=str)
    parser.add_argument('d', type=int, default=50)

    # Parameters for Least Squares
    parser.add_argument('--L', type=float, default=10000.0)
    parser.add_argument('--mu', type=float, default=1.0)

    # Parameters for Logistic
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--lam', type=float, default=1e-5)

    parser.add_argument("--T", type=int, default=10000, help="Number of function evaluations")
    parser.add_argument('--out_dir', type=str, default='./')
    parser.add_argument('--reps', type=int, default=50)

    parser.add_argument('--h', type=float, default=1e-7)
    args = parser.parse_args()
    run_experiment(args)