import sys

sys.path.append("../../targets/")
import time
import os
import torch
import numpy as np
from math import sqrt
from fdopt import ZD
from fdopt.grad_estimators import CentralGenStructGradApprox, CentralGenGradApprox, GenGradApprox, GenStructGradApprox, SinglePointEstimator, ForwardFD, CentralFD, LinInterpolation, CentralLinInterpolation
from fdopt.stepsize import PolyakStepsize, ConstantStepsize
from fdopt.direction_generators import GaussianDirections, SphericalDirections, CoordinateDirections, QRDirections

from synthetic import SphereFunction, LeastSquares
sys.path.append("../../")
from utils import ConvCriterion

device = 'cpu'
seed = 12131415
dtype = torch.float64

h_const = 1e-3
def build_ffd_optimizer(f, d, l, P_class):
    gamma = PolyakStepsize(gamma_0=l/d, f_min=0.0)
    h = ConstantStepsize(gamma_0=h_const)
    P = P_class(d = d, l = l, dtype = dtype, device = device)
    g = ForwardFD(d = d, l = l, f=f, P = P, normalize=False)
    return ZD(g = g, gamma = gamma, h = h)

def build_cfd_optimizer(f, d, l, P_class):
    gamma = PolyakStepsize(gamma_0=l/d, f_min=0.0)
    h = ConstantStepsize(gamma_0=h_const)
    P = P_class(d = d, l = l, dtype = dtype, device = device)
    g = CentralFD(d = d, l = l, f=f, P = P, normalize=False)
    return ZD(g = g, gamma = gamma, h = h)

def build_li_optimizer(f, d, l, P_class):
    if d != l:
        return None
    gamma = PolyakStepsize(gamma_0=l/d, f_min=0.0)
    h = ConstantStepsize(gamma_0=h_const)
    P = P_class(d = d, l = l, dtype = dtype, device = device)
    g = LinInterpolation(d = d, l = l, f=f, P = P, normalize=False)
    return ZD(g = g, gamma = gamma, h = h)

def build_gga_optimizer(f, d, l, P_class):
    gamma = PolyakStepsize(gamma_0=l/d, f_min=0.0)
    h = ConstantStepsize(gamma_0=h_const)
    P = P_class(d = d, l = l, dtype = dtype, device = device)
    g = GenGradApprox(d = d, l = l, f=f, P = P, normalize=False)
    return ZD(g = g, gamma = gamma, h = h)

def build_gsga_optimizer(f, d, l, P_class):
    gamma = PolyakStepsize(gamma_0=l/d, f_min=0.0)
    h = ConstantStepsize(gamma_0=h_const)
    P = P_class(d = l, l = l, dtype = dtype, device = device)
    g = GenStructGradApprox(d = d, l = l, f=f, P = P, normalize=False)
    return ZD(g = g, gamma = gamma, h = h)


def get_optimizer(opt_type):
    if opt_type == 'ffd':
        return build_ffd_optimizer    
    elif opt_type == 'cfd':
        return build_cfd_optimizer
    elif opt_type =='li':
        return build_li_optimizer
    elif opt_type == 'gga':
        return build_gga_optimizer
    elif opt_type == 'gsga':
        return build_gsga_optimizer

opt_type = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1

T = None
conv_thresholds = [1e-3, 1e-5, 1e-7]# [1e-1, 1e-3, 1e-5, 1e-7]
dimensions = [2, 4, 5, 6, 7, 10, 20, 50]#, 100, 500] #[2, 4, 6, 8, 10, 15, 20, 50, 100]
#n_directions = {
 #   2 : [1, 2],
 #   4 : [1, 2, 4],
 #   5 : [1, 2, 4, 5],
#    10 : [1, 5, 10],
#    20 : [1, 5, 10, 15, 20],
#    50 : [1, 5, 10, 25, 40, 50],
#    100 : [1, 25, 50, 75, 100],
#    500 : [1, 50, 100, 250, 400, 500],
#}
dim_dirs = [
    (3, [1, 2, 3]),
    (5, [1, 3, 4, 5]),
    (10, [1, 3, 5, 7, 10]),
#    (15, [1, 5, 7, 10, 15]),
#    (18, [1, 5, 7, 10, 15, 18]),
    (20, [1, 5, 7, 10, 15, 20]),
    (50, [1, 10, 25, 40, 50]),
#    (100, [1, 30, 70, 100]),
#    (200, [1, 50, 150, 200])
]


directions = [QRDirections]# [SphericalDirections, GaussianDirections, CoordinateDirections, QRDirections] # [GaussianDirections, SphericalDirections, CoordinateDirections, QRDirections]

opt_generator = get_optimizer(opt_type)
results = {}
for tau in conv_thresholds:
    conv_crit = ConvCriterion(tau = tau, f_L = 0.0)
    for (d, vals) in dim_dirs:
        for l in vals:
            for P_class in directions:
                f = SphereFunction(d = d, sigma = 0.0, seed = seed, device = device, dtype = dtype)
                x_0 = torch.ones((1, d), dtype=dtype, device=device)#torch.tensor([i for i in range(d)], dtype = dtype, device = device).reshape(1, d)

                opt = opt_generator(f = f, d = d, l = l, P_class = P_class)
                if opt is None:
                    continue
                results[f"{str(opt.g)}"] = {
                    'iterations' : [],
                    'values' : [],
                    'convergence' : [],
                    'elapsed_time' : []
                }
                path = f"../results/convex_benchmark/{repr(f)}/Polyak/{str(opt.g)}"
                os.makedirs(path, exist_ok=True)
                print(f"[--] Evaluating d = {d}\tl = {l}\tP : {P_class}...")
                with open(f"{path}/results_{tau}.log", 'a') as f:
                    for _ in range(reps):
                        it_time = time.time()
                        ris = opt.optimize(x_0 = x_0, T = T, convergence_criterion=conv_crit)
                        it_time = time.time() - it_time

                        results[f"{str(opt.g)}"]['iterations'].append(ris['iterations'])
                        results[f"{str(opt.g)}"]['values'].append(ris['fx'].item())
                        results[f"{str(opt.g)}"]['convergence'].append(ris['converged'])
                        results[f"{str(opt.g)}"]['elapsed_time'].append(it_time)
                        opt.reset()

                    f.write(f"{str(opt.g.P)}," + f"{reps}," + ",".join([str(x) for x in results[f"{str(opt.g)}"]['iterations']])
                            + ","  + ",".join([str(x) for x in results[f"{str(opt.g)}"]['values']])
                            + ","  + ",".join([str(int(x)) for x in results[f"{str(opt.g)}"]['convergence']])
                            + "\n")
                iterations = results[f"{str(opt.g)}"]['iterations']
                num_conv = np.sum(results[f"{str(opt.g)}"]['convergence'])
                it_time = results[f"{str(opt.g)}"]['elapsed_time']
                fvals = results[f"{str(opt.g)}"]['values']
                with open(f"{path}/{opt.g}.log", 'a') as f:
                    f.write(f"{tau},{np.mean(iterations)},{np.std(iterations)},{num_conv/reps},{np.mean(it_time)},{np.std(it_time)},{np.mean(fvals)},{np.std(fvals)}\n")
                print(f"{tau},{np.mean(iterations)},{np.std(iterations)},{num_conv/reps},{np.mean(it_time)},{np.std(it_time)},{np.mean(fvals)},{np.std(fvals)}\n")

#print(results)    



