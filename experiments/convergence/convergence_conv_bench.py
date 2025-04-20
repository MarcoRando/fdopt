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

#from synthetic_targets import LeastSquares, RegularizedLogistic
from convex_benchmark import *
from utils import compute_error_time, get_grad_estimator, get_target_function, get_grad_cost

methods = {
   'spherical' : SphericalDirections,
   'gaussian' : GaussianDirections,
   'rademacher' : RademacherDirections,
   'coordinate' : CoordinateDirections,
    'householder' : Householder,
    'butterfly' : ButterflyMatrix,
    'qr' : QRDirections,
}


grad_estimator = sys.argv[1]
base_dir = sys.argv[2]

#d = int(sys.argv[3])

sigma_noise = float(sys.argv[3])

#L = float(sys.argv[5])
#mu = float(sys.argv[6])

reps = int(sys.argv[4])

out_dir = f"{base_dir}/convergence/noise_{sigma_noise}/{grad_estimator}"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)




dtype = torch.float64
device = 'cuda'
seed = 12131419
h = 1e-7
functions = {
    'Matyas' : MatyasFunction(dtype=dtype, device=device, seed=seed),
    'Booth' : BoothFunction(dtype=dtype, device=device, seed=seed),
#    'Schwefel_2_23_100' : Schwefel223Function(d=100, dtype=dtype, device=device, seed=seed),
    'Schwefel_2_23_500' : Schwefel223Function(d=500, dtype=dtype, device=device, seed=seed),
    'Schwefel_2_23_1000' : Schwefel223Function(d=1000, dtype=dtype, device=device, seed=seed),
    # 'Zakharov_10' : ZakharovFunction(d = 10, dtype=dtype, device=device, seed=seed),
    # 'Zakharov_25' : ZakharovFunction(d = 25, dtype=dtype, device=device, seed=seed),
    # 'Zakharov_50' : ZakharovFunction(d = 50, dtype=dtype, device=device, seed=seed),
    
}



#fun = get_target_function(target_name='least_squares', d=d, mu=mu, L=L, set_l=True, device=device, dtype=dtype, sigma_noise=sigma_noise)

for (name, fun) in functions.items():

    d = fun.d # NOTE THAT IT IS DIFFERENT FROM PASSED PARAMETER CUTEST functions!
    g = get_grad_estimator(grad_estimator=grad_estimator, norm_const=1.0)
    T = 100
    print("[--] Function: {}".format(name))
    num_directions = [d] #[1, d//2, round((2/3) * d), round((4/5) * d), round((9/10) * d), d ]
    #num_directions = [d]# list(range(1, d, 2)) + [d]
    data = []
    for (i, l) in enumerate(num_directions):
        
        grad_cost = get_grad_cost(grad_estimator, l)
        print(f"[++] Computing for l = {l} and h = {h}")
        for (label, matrix_cls) in methods.items():
            generator = torch.Generator(device=device).manual_seed(123123123)
            print(f"[--] Computing for {label}")
#            P = matrix_cls(d = d, l = l, dct=True, dtype=dtype, device=device) if label == 'butterfly_dct' else  matrix_cls(d = d, l = l, dtype=dtype, device=device)
            P =  matrix_cls(d = d, l = l, dtype=dtype, device=device)
            g.nrm_const = 1/l if label=='gaussian' or label=='rademacher' else d/l
            gamma = 1.0 #PolyakStep(gamma_0=1.0, eps_k=lambda t: 1/(t + 1), approx_min=True)
#            opt = FDOpt(P = P, g = g, gamma = 0.001, h = h)
            opt = LSFDOpt(P = P, g = g, gamma_0=gamma, h = h, grad_cost=grad_cost, gamma_minmax=(1e-10, 1000.0), theta=0.5, rho=2.0, c = 0.01)
            X = (fun.bounds[:, 1] - fun.bounds[:, 0]).to(device) * torch.rand((reps, d), dtype=dtype, device=device, generator=generator) + fun.bounds[:, 0].to(device)
 #           class_errors = []
            for r in range(reps):
                x0 =  X[r, :].reshape(1, -1).clone()
                f_0 = fun(x0).cpu().item()

                x, values, num_evals, gammas, grad_norms = opt.optimize(fun, x_0 =x0, T = T)
                with open(f"{out_dir}/{name}_{label}_{h}_{r}.csv", 'w') as f:
                    for i in range(len(values)):
                        f.write("{},{},{},{},{}\n".format(f_0, values[i], num_evals[i], gammas[i], grad_norms[i]))
                        f.flush()
#                class_errors.append((( (fun.X @ x.flatten()).sign() != fun.y).to(dtype).sum().cpu().item() / n) * 100)
#            print("Error perc: {} +/- {}".format(np.mean(class_errors), np.std(class_errors) ) )
