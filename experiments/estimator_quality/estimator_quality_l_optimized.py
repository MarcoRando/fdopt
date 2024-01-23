import os
import sys
import torch
import time
import numpy as np
from math import sqrt
from fdopt.surrogates import ForwardFD, CentralFD, OptimizedForwardFD
from fdopt.direction_generators import Householder, ButterflyMatrix, CoordinateDirections, GaussianDirections, SphericalDirections, QRDirections

from synthetic_targets import LeastSquares, Rosenbrock, Rastrigin, StyblinskiTang, Trid, SumOfSquares
from utils import relative_error

methods = {
#    'coordinate' : CoordinateDirections,
    # 'coordinate' : CoordinateDirections,
    # 'spherical' : SphericalDirections,
    # 'householder' : Householder,
    # 'butterfly' : ButterflyMatrix,
    # 'butterfly_dct' : ButterflyMatrix,
    # 'qr' : QRDirections,
    'optimized_gaussian' : GaussianDirections,
    'optimized_spherical' : SphericalDirections,
    'optimized_householder' : Householder,
    'optimized_butterfly' : ButterflyMatrix,
    'optimized_butterfly_dct' : ButterflyMatrix,
    'optimized_qr' : QRDirections
}
h = 1e-7

def compute_error_time(fun, x, g, P, reps = 10):
    errors, times = [], []
    grad = fun.grad(x).flatten()
    _ = g(fun, x, h = h, P = P())
    for _ in range(reps):
        it_time = time.time()
        g_x = g(fun, x, h = h, P = P())
        it_time = time.time() - it_time
        rel_err = relative_error(g_x, grad).item()
        errors.append(rel_err)
        times.append(it_time)
    return np.mean(errors), np.std(errors), np.mean(times), np.std(times)

#mu, L = int(sys.argv[1]), int(sys.argv[2])

out_dir = "./results/changing_l/optimized"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)

#dimensions = [2, 4, 8, 10, 20, 50, 100, 200, 400, 500, 1000]#, 2000]

dimensions = {
    8 : [2, 4, 6, 7, 8],
    20 : [2, 5, 10, 15, 18, 19, 20],
    50 : [2, 10, 25, 40, 48, 49, 50],
    100 : [2, 40, 50, 60, 80, 98, 99, 100],
    400 : [2, 50, 100, 150, 200, 300, 397, 398, 399, 400],
    500 : [100, 200, 250, 300, 400, 450, 498, 499, 500],
    1000 : [200, 300, 400, 450, 500, 550, 700, 800, 900, 998, 999, 1000],
}

generator = torch.Generator()
generator.manual_seed(12131415)

functions = [
    ('Rosenbrock', Rosenbrock), 
    ('Rastrigin', Rastrigin), 
    ('StyblinskiTang', StyblinskiTang), 
    ('Trid',Trid), 
    ('SumOfSquares',SumOfSquares),
    ('LeastSquares', LeastSquares)
    ]

g = ForwardFD(vectorized=True)
g_opt = OptimizedForwardFD(vectorized=True)
for (fun_name, fun_cls) in functions:
#    fun_name = "LeastSquares"

    dtype = torch.float64
    device = 'cpu'
    seed = 12131415
    max_val, min_val = 10.0, -10.0
    reps=100
    for (i, (d, l_list)) in enumerate(dimensions.items()):
        # LeastSquares, Rosenbrock, Rastrigin, StyblinskiTang, Trid, SumOfSquares
        #fun = SumOfSquares(d = d, dtype=dtype, device=device, seed = seed)
        if fun_name == 'LeastSquares':
            fun = LeastSquares(mu=1, L=10, d=d, dtype=torch.float64)
        else:
            fun = fun_cls(d = d, dtype=dtype, device=device, seed=seed)
        for l in l_list:
            print(f"[--] Computing for d = {d} l = {l}")
            with open(f"{out_dir}/{fun_name}_{d}_{l}.log", 'a') as f:
                for (label, matrix_cls) in methods.items():
                    if label == 'butterfly_dct':
                        P = matrix_cls(d = d, l =l, dct=True, dtype=torch.float64)
                    else:
                        P = matrix_cls(d = d, l =l, dtype=torch.float64)
                    x = (max_val - min_val) * torch.rand((1, d), generator = generator, dtype=torch.float64) + min_val
                    if 'optimized' in label: 
                        mu_err, std_err, mu_time, std_time = compute_error_time(fun, x, g_opt, P, reps = reps)
                    else:
                        mu_err, std_err, mu_time, std_time = compute_error_time(fun, x, g, P, reps = reps)
                    f.write("{},{},{},{},{}\n".format(label,mu_err, std_err, mu_time, std_time))
                    f.flush()