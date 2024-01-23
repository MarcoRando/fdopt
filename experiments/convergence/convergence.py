import os
import torch
import numpy as np
from synthetic_targets import SumOfSquares, LeastSquares, L1Norm

from fdopt.direction_generators import *
from fdopt.surrogates import ForwardFD, CentralFD, OptimizedForwardFD
from fdopt.optimizers import LSZD


def optimize_test(label, opt, x0, f_star, eps, T, reps = 10):
    evaluations = []
    for r in range(reps):
        num_evals = opt.optimize(x0, f_star=f_star, eps=eps, T=T, verbose=False)
        evaluations.append(num_evals)
        print("[{}] Completed iteration {} with {} function evals".format(label, r, num_evals))
    return np.mean(evaluations), np.std(evaluations)
        
d = 1000
l = 1000
dtype = torch.float64
target = L1Norm(d = d, dtype=dtype)
g = ForwardFD(vectorized=True)
h = lambda t : 1e-7
gamma_0 = 1.0
armijo_constant = 0.1
theta = 0.3
rho = 2.0
eps=1e-3
min_val, max_val = -5, 5
T = 100000
x_0 = (max_val - min_val) * torch.rand(1, d, dtype=dtype) + min_val

out_dir = "./results/ffd"
os.makedirs(out_dir, exist_ok=True)
fname="L1Norm"
directions = [
    ('householder', Householder(d=d, l=l, dtype=dtype)),
    ('butterfly', ButterflyMatrix(d=d, l=l,dct=True, dtype=dtype)),
    ('coordinate', CoordinateDirections(d=d, l=l, dtype=dtype, store_identity=True)),
    ('spherical', SphericalDirections(d=d, l=l, dtype=dtype)),
    ('qr', QRDirections(d=d,l=l, dtype=dtype))
    ]

for (label, P) in directions:
    opt = LSZD(target=target,
            g = g, 
            P = P, 
            h=h, gamma_0=gamma_0, theta=theta, rho=rho, armijo_constant=armijo_constant)
    mu_evals, std_evals = optimize_test(label, opt, x_0, f_star=0.0, eps=1e-3, T=T, reps=5)
    with open(f"{out_dir}/{fname}_{d}_{l}.log", 'a') as f:
        f.write("{},{},{}\n".format(label, mu_evals,std_evals)) 

