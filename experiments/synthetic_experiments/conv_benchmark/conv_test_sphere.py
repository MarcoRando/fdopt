import sys
from typing import Any

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


from synthetic import SphereFunction


class ConvCriterion:
    def __init__(self, tau):
        self.tau = tau
        self.ratios = []
        
    def __call__(self, k, g_k, f_k, f_0) -> Any:
        ratio = ((f_k - f_L) / (f_0 - f_L) ).item()
        self.ratios.append(ratio)
        return ratio <= tau

dtype = torch.float64
device = 'cpu'

f_L = 0.0
taus = [1e-3, 1e-4, 1e-5]#, 1e-8]

d = 100
num_iters = 50
num_split = 10
budget = 1000 #num_iters * l * num_split

for P_class in [GaussianDirections, SphericalDirections, RademacherDirections, CoordinateDirections, QRDirections]:
    for l in [30, 50, 70]:#range(1, d + 1):
        target = SphereFunction(d = d, dtype = dtype, device=device)

        gamma = ConstantStepsize(gamma_0=None)
        P = P_class(d = d, l = l, dtype=dtype, device=device)
        g = ForwardFD(d =d, l = l, f = target, P = P)
        zd = ZD(g = g, gamma=gamma, h = ConstantStepsize(gamma_0=1e-7))
        pfzd = PFZD(zd)
        print(f"[{g}_{P}] d = {d}\tl = {l}\tbudget = {budget}")
        x0 = torch.tensor([i for i in range(d)]).reshape(1, d)
        T = 1000
        path = f"../results/convex_benchmark/{target.name}/"
        os.makedirs(path, exist_ok=True)
        reps = 100
        for tau in taus:
            n_iters = []
            opt_times = []
            num_conv = 0
            for _ in range(reps):
                crit = ConvCriterion(tau)
                o_time = time.time()
                ris = pfzd.optimize(x_0=x0.clone(), T = budget, 
                                    gamma_space=(1e-10,1.0), 
                                    num_split=num_split,return_trace=True, 
                                    convergence_criterion=crit)
                o_time = time.time() - o_time
                if ris['converged']:
                    num_conv += 1
                    opt_times.append(o_time)
                    n_iters.append(ris['iterations'])
            with open(f"{path}/{g}_{P}_{budget}.log", 'a') as f:
                f.write(f"{tau},{np.mean(n_iters)},{np.std(n_iters)},{num_conv/reps},{np.mean(opt_times)},{np.std(opt_times)}\n")
            print(ris['fx'].item(), ris['converged'], ris['iterations'])

