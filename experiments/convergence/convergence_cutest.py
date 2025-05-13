import os
import sys
import argparse as ap
import torch
import sys
import time
import pandas as pd
import numpy as np
from math import sqrt
import pycutest as pc 

from fdopt.direction_generators import * #Householder, ButterflyMatrix, CoordinateDirections, GaussianDirections, SphericalDirections, QRDirections
from fdopt.optimizers import FDOpt, LSFDOpt
from fdopt.grad_estimators import ForwardFD

sys.path.append("../")

from synthetic_targets import LeastSquares, PycutestTarget
from utils import compute_error_time, get_grad_estimator, get_target_function


methods = {
    'spherical' : SphericalDirections,
    'gaussian' : GaussianDirections,
    'rademacher' : RademacherDirections,
    'coordinate' : CoordinateDirections,
    'householder' : Householder,
    "permuted_householder" : PermutedHouseholder,
    'butterfly' : ButterflyMatrix,
    'qr' : QRDirections,
}

def get_pycutest_target(problem_name):
    
    splitted = problem_name.split(',')
    
    name = splitted[0]
    d = int(splitted[1])
    n = int(splitted[2])
    
    x = np.array([float(elem) for elem in splitted[3:]])
    
    problem = pc.import_problem(name.split("_")[0], sifParams={'N' : d})
    return PycutestTarget(problem, dtype=dtype, device=device, seed = seed, x0=x), d, name


dtype = torch.float64
device = 'cuda'
seed = 12131419

def run_experiment(args):


    base_dir = args.out_dir

    sigma_noise = args.sigma

    T = args.T
    h = args.h
    reps = args.reps 

    out_dir = f"{base_dir}/convergence/noise_{sigma_noise}/ffd/cutest"#.format(mu, L)

    os.makedirs(out_dir, exist_ok=True)
    with open("../cutest_problem_names.log", 'r') as f:
        problems = f.readlines()

    data = []
    for problem in problems:
        g = ForwardFD(norm_const=1.0) # get_grad_estimator(grad_estimator=grad_estimator, norm_const=1.0)
        fun, d_idx, problem_name = get_pycutest_target(problem)
        d = fun.d
        num_directions = [2, d//3, d//2, round((4/5) * d),  d]
        num_directions = num_directions[::-1]
        for (i, l) in enumerate(num_directions):
            
            print(f"[{problem}] Computing for d = {d} l = {l} and h = {h}")
            for (label, matrix_cls) in methods.items():
                generator = torch.Generator(device=device).manual_seed(123123123)
                print(f"[--] Computing for {label}")
                P = matrix_cls(d = d, l = l, dtype=dtype, device=device)
                g.nrm_const = 1/l if label=='gaussian' or label=='rademacher' else d/l
                gamma = 0.5 
                opt = LSFDOpt(P = P, g = g, gamma_0=gamma, h = h, gamma_minmax=(1e-10, 1.0), theta=0.5, rho=2.0, c = 0.00001)
                x0 = fun.x0
                f_0 = fun(x0).cpu().item()
                for r in range(reps):
                    _, values, num_evals, gammas, it_times = opt.optimize(fun, x_0 = x0, T = T)
                    with open(f"{out_dir}/{problem_name}_{d_idx}_{d}_{label}_{l}_{h}_{r}.csv", 'w') as f:
                        for i in range(len(values)):
                            f.write("{},{},{},{},{},{}\n".format(f_0, min(values), values[i], num_evals[i], gammas[i], it_times[i]))
                            f.flush()


if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Compute gradient approximation error for a given target.", formatter_class=ap.ArgumentDefaultsHelpFormatter)


    parser.add_argument("--T", type=int, default=10000, help="Number of function evaluations")
    parser.add_argument('--out_dir', type=str, default='./')
    parser.add_argument('--reps', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.0)

    parser.add_argument('--h', type=float, default=1e-7)
    args = parser.parse_args()
    run_experiment(args)