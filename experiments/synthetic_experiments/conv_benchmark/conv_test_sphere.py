import sys

sys.path.append("../../targets/")
import time
import os
import torch
import numpy as np
from math import sqrt


from fdopt import ZD, PFZD
from fdopt.grad_estimators import CentralGenStructGradApprox, CentralGenGradApprox, GenGradApprox, GenStructGradApprox, SinglePointEstimator, ForwardFD, CentralFD, LinInterpolation, CentralLinInterpolation
from fdopt.stepsize import ConstantStepsize, Stepsize
from fdopt.direction_generators import GaussianDirections, RademacherDirections, SphericalDirections, CoordinateDirections, QRDirections


from synthetic import SphereFunction, Quadratic, LeastSquares

sys.path.append("../../")
from utils import ConvCriterion

dtype = torch.float64
device = 'cpu'

f_L = 0.0
taus = [1e-3, 1e-5, 1e-7]#, 1e-8]

num_split = 20
budget = 10000 # num_iters * l * num_split
reps = 100

# l_vals = [1, 2, 3] # d= 3
approximations = [ForwardFD]#[ForwardFD]# [ForwardFD, CentralFD, GenGradApprox, GenStructGradApprox]
#d = 200
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

for app in approximations:
    for (d, l_vals) in dim_dirs:
        for P_class in [CoordinateDirections, QRDirections]:#[GaussianDirections, SphericalDirections, CoordinateDirections, QRDirections]:
            for l in l_vals:#range(1, d + 1):
                target = SphereFunction(d = d, dtype = dtype, device=device)
               # target = Quadratic(d = d, dtype = dtype, device=device)
        #        target = LeastSquares(d = d, m = d, mu = 4.0, L=5.0, dtype = dtype, device=device)

                gamma = ConstantStepsize(gamma_0=None)
                if issubclass(app, GenStructGradApprox):
                    P = P_class(d = l, l = l, dtype=dtype, device=device)
                else:
                    P = P_class(d = d, l = l, dtype=dtype, device=device)
                g = app(d =d, l = l, f = target, P = P)
                zd = ZD(g = g, gamma=gamma, h = ConstantStepsize(gamma_0=1e-5))
                pfzd = PFZD(zd)
                print(f"[{g}] d = {d}\tl = {l}\tbudget = {budget}")
                x0 = torch.tensor([i for i in range(d)], dtype=dtype).reshape(1, d)
                path = f"../results/convex_benchmark/{repr(target)}/Bisection/{str(zd.g)}"
                os.makedirs(path, exist_ok=True)
                for tau in taus:
                    n_iters = []
                    opt_times = []
                    num_conv = 0
                    fvals = []
                    for _ in range(reps):
                        crit = ConvCriterion(tau, f_L=f_L)
                        o_time = time.time()
                        ris = pfzd.optimize(x_0=x0.clone(), T = budget, 
                                            gamma_space=(1e-7,1e-1), 
                                            num_split=num_split,return_trace=True, 
                                            convergence_criterion=crit)
                        o_time = time.time() - o_time
                        fvals.append(ris['fx'].item())
                        if ris['converged']:
                            num_conv += 1
                            opt_times.append(o_time)
                            n_iters.append(ris['iterations'])
                    with open(f"{path}/{g}_{budget}.log", 'a') as f:
                        f.write(f"{tau},{np.mean(n_iters)},{np.std(n_iters)},{num_conv/reps},{np.mean(opt_times)},{np.std(opt_times)},{np.mean(fvals)},{np.std(fvals)}\n")
                    print(f"{tau},{np.mean(n_iters)},{np.std(n_iters)},{num_conv/reps},{np.mean(opt_times)},{np.std(opt_times)},{np.mean(fvals)},{np.std(fvals)}")
                
