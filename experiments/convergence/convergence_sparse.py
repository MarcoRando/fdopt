import os
import torch
import time
import numpy as np
from synthetic_targets import SumOfSquares, LeastSquares, L1Norm, SparseSOS

from fdopt.direction_generators import *
from fdopt.surrogates import ForwardFD, CentralFD, OptimizedForwardFD
from fdopt.optimizers import LSZD, PFZD


def optimize_test(label, opt, x0, f_star, eps, T, reps = 10):
    evaluations = []
    times = []
    for r in range(reps):
        it_time = time.time()
        num_evals = opt.optimize(x0, f_star=f_star, eps=eps, T=T, verbose=True)
        it_time = time.time() - it_time
        evaluations.append(num_evals)
        times.append(it_time)
        print("[{}] Completed iteration {} with {} function evals".format(label, r, num_evals))
    return np.mean(evaluations), np.std(evaluations), np.mean(times), np.std(times)


#dimensions = [10, 20, 30, 50, 100, 200, 250, 500, 1000]

torch.manual_seed(12131415)

num_directions = {
    10 : [1, 2, 5, 9, 10],
    30 : [1, 5, 10, 15, 20, 25, 29, 30]
}

num_components = {
    10 : [1, 2, 5, 9, 10],
    30 : [1, 5, 10, 15, 20, 25, 29, 30]    
}
dimensions = [1000]#[10, 20, 30, 50, 100, 200, 250, 500, 1000]
l = 10
num_comp = 10
max_gamma = 1.0
min_gamma = 1e-5
num_splits = 10
eps=1e-3

for d in dimensions:  
    dtype = torch.float64
    target = SparseSOS(num_components = num_comp, d = d, dtype=dtype)
    g = ForwardFD(vectorized=True)
    g_opt = OptimizedForwardFD()
    h = lambda t : 1e-9
    min_val, max_val = -1, 1
    T = 100000
    x_0 = (max_val - min_val) * torch.rand(1, d, dtype=dtype) + min_val

    out_dir = "./results/sparse/ffd"
    os.makedirs(out_dir, exist_ok=True)
    fname="SparseSOS"
    directions = [
        ('coordinate', CoordinateDirections(d=d, l=l, dtype=dtype, store_identity=True)),
        ('spherical', SphericalDirections(d=d, l=l, dtype=dtype)),
        ('qr', QRDirections(d=d,l=l, dtype=dtype)),
        ('optimized_spherical', SphericalDirections(d=d, l=l, dtype=dtype)),
        ('optimized_gaussian', GaussianDirections(d=d, l=l, dtype=dtype)),
        ('butterfly', ButterflyMatrix(d=d, l=l,dct=True, dtype=dtype)),
        ('householder', Householder(d=d, l=l, dtype=dtype)),
    #    ('gaussian', GaussianDirections(d=d, l=l, dtype=dtype)),
        ]

    for (label, P) in directions:
        if 'optimized' in label:
            opt = PFZD(target=target,
                    g = g_opt, 
                    P = P,
                    h=h, num_splits = num_splits,
                    max_gamma = max_gamma, min_gamma=min_gamma)        
        else:
            opt = PFZD(target=target,
                    g = g, num_splits = num_splits,
                    P = P, max_gamma = max_gamma, min_gamma=min_gamma,
                    h=h)
        mu_evals, std_evals, mu_tms, std_tms = optimize_test(label, opt, x_0, f_star=0.0, eps=1e-3, T=T, reps=5)
        with open(f"{out_dir}/{fname}_{d}_{l}_{num_comp}.log", 'a') as f:
            f.write("{},{},{},{},{}\n".format(label, mu_evals, std_evals, mu_tms, std_tms)) 
            f.flush()
