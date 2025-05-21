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
from fdopt.grad_estimators import ForwardFD #, CentralFD
from fdopt.direction_generators import *# Householder, ButterflyMatrix, CoordinateDirections, GaussianDirections, SphericalDirections, QRDirections, SVDDirections


sys.path.append("../")

from synthetic_targets import LeastSquares, PycutestTarget
from utils import compute_error_time, get_target_function

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

dtype = torch.float64
device = 'cuda'
seed = 12131419


def get_pycutest_target(problem_name):
    
    splitted = problem_name.split(',')
    
    name = splitted[0]
    d = int(splitted[1])
    n = int(splitted[2])
    
    x = np.array([float(elem) for elem in splitted[3:]])
    
    problem = pc.import_problem(name.split("_")[0], sifParams={'N' : d})
    return PycutestTarget(problem, dtype=dtype, device=device, seed = seed, x0=x), d


def run_experiment(args):
    base_dir = args.out_dir


    h = args.h
    reps = args.reps 

    out_dir = f"{base_dir}/grad_approx_error/ffd/cutest"#.format(mu, L)

    os.makedirs(out_dir, exist_ok=True)

    with open("../cutest_problem_names.log", 'r') as f:
        problems = f.readlines()

    for problem in problems:
        g = ForwardFD(norm_const=1.0) 
        fun, d_idx = get_pycutest_target(problem)
        
        x = fun.x0 
        d = x.shape[1]
        num_directions = [2, d//3, d//2, round((4/5) * d), d] 
        data = []
        for (j, l) in enumerate(num_directions):
            prob_name = problem.split(",")[0]
            print(f"[++] Estimating gradient of {prob_name} [{d_idx}]  using {l} of {d} directions")
            for (label, matrix_cls) in methods.items():
                P = matrix_cls(d = d, l = l, dtype=dtype, device=device)
                g.nrm_const = 1/l if label=='gaussian' or label =='rademacher' else d/l
                
                errors, times = compute_error_time(fun, x.clone(), g, P, h, reps = reps)
                for i in range(len(errors)):
                    data.append([problem.strip().split(',')[0], d, label, j, l, errors[i], times[i], i]) 

                print(f"\t[--] Completed for {label}")
        dframe = pd.DataFrame(data, columns=['function', 'dimension', 'directions', 'l_idx', 'l', 'err', 'time', 'rep'])
        dframe.to_csv(f"{out_dir}/{problem.strip().split(',')[0]}_{d_idx}_{d}_{h}.csv", index=False)
        fun.close()


if __name__ == '__main__':
    parser = ap.ArgumentParser(description="Compute gradient approximation error for CUTEST functions.", formatter_class=ap.ArgumentDefaultsHelpFormatter)



    parser.add_argument('--out_dir', type=str, default='./')
    parser.add_argument('--reps', type=int, default=50)

    parser.add_argument('--h', type=float, default=1e-7)
    args = parser.parse_args()
    run_experiment(args)