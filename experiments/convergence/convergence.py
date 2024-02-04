import os
import torch
import time
import numpy as np
from synthetic_targets import SumOfSquares, LeastSquares, L1Norm

from fdopt.direction_generators import *
from fdopt.surrogates import ForwardFD, CentralFD, OptimizedForwardFD
from fdopt.optimizers import LSZD


def optimize_test(label, opt, x0, f_star, eps, T, reps = 10):
    evaluations = []
    times = []
    for r in range(reps):
        it_time = time.time()
        num_evals = opt.optimize(x0, f_star=f_star, eps=eps, T=T, verbose=False)
        it_time = time.time() - it_time
        evaluations.append(num_evals)
        times.append(it_time)
        print("[{}] Completed iteration {} with {} function evals".format(label, r, num_evals))
    return np.mean(evaluations), np.std(evaluations), np.mean(times), np.std(times)
        
d = 1000
l = 1000
dtype = torch.float64
target = LeastSquares(d = d, L=10, mu=1, dtype=dtype)
g = ForwardFD(vectorized=True)
g_opt = OptimizedForwardFD()
h = lambda t : 1e-9
gamma_0 = 1.0
armijo_constant = 0.1
theta = 0.5
rho = 2.0
eps=1e-3
min_val, max_val = -5, 5
T = 100000
x_0 = (max_val - min_val) * torch.rand(1, d, dtype=dtype) + min_val

out_dir = "./results/ffd"
os.makedirs(out_dir, exist_ok=True)
fname="LeastSquares"
directions = [
    ('householder', Householder(d=d, l=l, dtype=dtype)),
    ('butterfly', ButterflyMatrix(d=d, l=l,dct=True, dtype=dtype)),
    ('coordinate', CoordinateDirections(d=d, l=l, dtype=dtype, store_identity=True)),
    ('spherical', SphericalDirections(d=d, l=l, dtype=dtype)),
    ('qr', QRDirections(d=d,l=l, dtype=dtype)),
    ('optimized_spherical', SphericalDirections(d=d, l=l, dtype=dtype)),
    ('optimized_gaussian', GaussianDirections(d=d, l=l, dtype=dtype)),
    ('gaussian', GaussianDirections(d=d, l=l, dtype=dtype)),
    ]

for (label, P) in directions:
    if 'optimized' in label:
        opt = LSZD(target=target,
                g = g_opt, 
                P = P, 
                h=h, gamma_0=gamma_0, theta=theta, rho=rho, armijo_constant=armijo_constant)        
    else:
        opt = LSZD(target=target,
                g = g, 
                P = P, 
                h=h, gamma_0=gamma_0, theta=theta, rho=rho, armijo_constant=armijo_constant)
    mu_evals, std_evals, mu_tms, std_tms = optimize_test(label, opt, x_0, f_star=0.0, eps=1e-3, T=T, reps=5)
    with open(f"{out_dir}/{fname}_{d}_{l}.log", 'a') as f:
        f.write("{},{},{},{},{}\n".format(label, mu_evals, std_evals, mu_tms, std_tms)) 

