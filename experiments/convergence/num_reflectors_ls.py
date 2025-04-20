import os
import sys
import argparse
import torch
import sys
import time
import pandas as pd
import numpy as np
from math import sqrt
from fdopt.direction_generators import * #Householder, ButterflyMatrix, CoordinateDirections, GaussianDirections, SphericalDirections, QRDirections
from fdopt.optimizers import FDOpt, LSFDOpt

sys.path.append("../")

from synthetic_targets import LeastSquares
from utils import compute_error_time, get_grad_estimator, get_target_function, get_grad_cost

methods = {
#    'spherical' : SphericalDirections,
#    'gaussian' : GaussianDirections,
#    'rademacher' : RademacherDirections,
#    'coordinate' : CoordinateDirections,
    'householder' : Householder,
#    'butterfly' : ButterflyMatrix,
#    'butterfly_dct' : ButterflyMatrix,
#    'qr' : QRDirections,
}


grad_estimator = sys.argv[1]
base_dir = sys.argv[2]

d = int(sys.argv[3])

sigma_noise = float(sys.argv[4])

L = float(sys.argv[5])
mu = float(sys.argv[6])

reps = int(sys.argv[7])

out_dir = f"{base_dir}/convergence/noise_{sigma_noise}/{grad_estimator}"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)




dtype = torch.float64
device = 'cuda'
seed = 12131419


#h = 5e-1

fun = LeastSquares(mu = mu, L = L, set_l = True, d=d,device=device, dtype=dtype, sigma_noise=sigma_noise)# get_target_function(fun_name, d=d, sigma_noise=sigma_noise, dtype=dtype, device=device) 

#fun = get_target_function(target_name='least_squares', d=d, mu=mu, L=L, set_l=True, device=device, dtype=dtype, sigma_noise=sigma_noise)

d = fun.d # NOTE THAT IT IS DIFFERENT FROM PASSED PARAMETER CUTEST functions!
x = fun.x0
g = get_grad_estimator(grad_estimator=grad_estimator, norm_const=1.0)
T = 500

num_directions = [d//2]# [1, d//2, round((2/3) * d), round((4/5) * d), round((9/10) * d), d ]

num_reflectors = [1, 2, 5, 10, 25, 35, 40, 50]
#num_directions = [d]# list(range(1, d, 2)) + [d]
f_0 = fun(fun.x0).cpu().item()
for h in [1e-7]: #[1e-1, 1e-3, 1e-5]:#[1e-1, 1e-3, 1e-7]:
    data = []
    for (i, l) in enumerate(num_directions):
        
        grad_cost = get_grad_cost(grad_estimator, l)
        fx = fun(x)
        print(f"[++] Computing for l = {l} and h = {h}")
        for n_ref in num_reflectors:
            print(f"[--] Computing for Num Reflectors = {n_ref}")
            P = Householder(d = d, l = l, num_reflectors=n_ref, dtype=dtype, device=device)
            g.nrm_const = d/l
            gamma = 0.5 #PolyakStep(gamma_0=1.0, eps_k=lambda t: 1/(t + 1), approx_min=True)
#            opt = FDOpt(P = P, g = g, gamma = 0.001, h = h)
            opt = LSFDOpt(P = P, g = g, gamma_0=gamma, h = h, grad_cost=grad_cost, gamma_minmax=(1e-10, 1.0), theta=0.5, rho=2.0, c = 0.0001)
            for r in range(reps):
                _, values, num_evals, gammas, grad_norms = opt.optimize(fun, x_0 = fun.x0, T = T)
                with open(f"{out_dir}/LeastSquare_NumRef_{n_ref}_{l}_{L}_{mu}_{d}_{h}_{r}.csv", 'w') as f:
                    for i in range(len(values)):
                        f.write("{},{},{},{},{}\n".format(f_0, values[i], num_evals[i], gammas[i], grad_norms[i]))
                        f.flush()
            print(values[-1] / f_0)
#            exit()
#            mu_err, std_err, mu_time, std_time = compute_error_time(fun, x.clone(), g, P, h, reps = reps)
#            errors, times = compute_error_time(fun, x.clone(), g, P, h, reps = reps)
#            for j in range(len(errors)):
 #               data.append([label, l, errors[j], times[j], j]) 
 #           print(f"\t[--] Completed for {label}")

#    dframe = pd.DataFrame(data, columns=['directions', 'l', 'err', 'time', 'rep'])
#    dframe.to_csv(f"{out_dir}/LeastSquare_{L}_{mu}_{d}_{h}.csv", index=False)
