import os
import sys
import torch
import time
import numpy as np
from math import sqrt
from fdopt.surrogates import ForwardFD, CentralFD
from fdopt.direction_generators import Householder, ButterflyMatrix, CoordinateDirections, GaussianDirections, SphericalDirections, QRDirections

from synthetic_targets import LeastSquares, Rosenbrock, Rastrigin, StyblinskiTang, Trid, SumOfSquares
from utils import relative_error

methods = {
    'coordinate' : CoordinateDirections,
#    'gaussian' : GaussianDirections,
    'spherical' : SphericalDirections,
    'householder' : Householder,
    'butterfly' : ButterflyMatrix,
    'butterfly_dct' : ButterflyMatrix,
    'qr' : QRDirections
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

out_dir = "./results/changing_d/cfd"#.format(mu, L)

os.makedirs(out_dir, exist_ok=True)

dimensions = [2, 4, 8, 10, 20, 50, 100, 200, 400, 500, 1000]#, 2000]
generator = torch.Generator()
generator.manual_seed(12131415)



g = CentralFD(vectorized=True)

fun_name = "Rosenbrock"

dtype = torch.float64
device = 'cpu'
seed = 12131415
max_val, min_val = 10.0, -10.0
reps=100
for (i, d) in enumerate(dimensions):
    #LeastSquares, Rosenbrock, Rastrigin, StyblinskiTang, Trid, SumOfSquares
    fun = Rosenbrock(d = d, dtype=dtype, device=device, seed = seed)
#    fun = SumOfSquares(d=d, dtype=dtype)
  #  fun = LeastSquares(mu=1, L=10, d=d, dtype=torch.float64)
    print(f"[--] Computing for d = {d}")
    with open(f"{out_dir}/{fun_name}_{d}.log", 'w') as f:
        for (label, matrix_cls) in methods.items():
            if label == 'butterfly_dct':
                P = matrix_cls(d = d, l =d, dct=True, dtype=torch.float64)
            else:
                P = matrix_cls(d = d, l =d, dtype=torch.float64)
            x = (max_val - min_val) * torch.rand((1, d), generator = generator, dtype=torch.float64) + min_val
            mu_err, std_err, mu_time, std_time = compute_error_time(fun, x, g, P, reps = reps)
            f.write("{},{},{},{},{}\n".format(label,mu_err, std_err, mu_time, std_time))
            f.flush()