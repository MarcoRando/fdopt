import torch
import numpy as np
import pycutest as pct
import time
from math import sqrt


from fdopt import ZD
from fdopt.grad_estimators import CentralGenStructGradApprox, CentralGenGradApprox, GenGradApprox, GenStructGradApprox, SinglePointEstimator, ForwardFD, CentralFD, LinInterpolation, CentralLinInterpolation
from fdopt.stepsize import ConstantStepsize, Stepsize
from fdopt.direction_generators import GaussianDirections, RademacherDirections, SphericalDirections, CoordinateDirections, QRDirections

d = 10
c = 0
dtype = torch.float64
device = 'cpu'
seed = 12121314
problems = pct.find_problems(objective='sum of squares', regular=True, userN = True)
pct.print_available_sif_params(problems[c])
target = pct.import_problem(problems[c], sifParams={'N' : d})
print(target.x0.shape)

def f(x):
    x = x.reshape(-1, d)
    results = torch.empty((x.shape[0], 1), dtype=dtype, device=device)
    for i in range(x.shape[0]):
        results[i, 0] = target.obj(x[i])
    return results


h_con = 1e-7

num_dirs = [i  for i in range(1, d + 1)]
h = ConstantStepsize(h_con)
budgets = [50, 100, 200, 500, 1000]
methods = [ForwardFD, GenStructGradApprox, SinglePointEstimator, CentralGenStructGradApprox, CentralGenGradApprox, GenGradApprox, CentralFD]
dirs = [GaussianDirections, RademacherDirections, SphericalDirections, CoordinateDirections, QRDirections]
iters = 100
gen = torch.Generator(device=device)
with open("./constant_stepsizes", 'a') as fl:
    for method in methods:
        for T in budgets:
            for l in num_dirs:
                for dir_mat in dirs:
                    x0 = torch.from_numpy(target.x0).reshape(1, d) #+ torch.randn(1, d, device=device, dtype=dtype, generator=gen)
                    if issubclass(method, CentralGenStructGradApprox) or issubclass(method, GenStructGradApprox):
                        if not (issubclass(dir_mat, CoordinateDirections) or  issubclass(dir_mat, QRDirections)):
                            continue
                        P = dir_mat(d = l, l = l, device=device, dtype=dtype, seed=seed)
                    else:
                        P = dir_mat(d = d, l = l, device=device, dtype=dtype, seed=seed)
                    g = method(d = d, l = l, f = f, P = P, normalize=True)
                    #(self, g : GradEstimator, gamma : Stepsize, h: Stepsize) -> None:
                    g_min, g_max = 1e-10, 1e-1
                    f0 = f(x0)
                    c_val = f0
                    best_gm, best_fx, exec_tm = None, None, None
                    for i in range(iters):
                        gm = np.mean([g_min, g_max])
                        gamma = ConstantStepsize(gm)
                        opt = ZD(g = g, gamma = gamma, h = h)
                        o_time = time.time()
                        ris = opt.optimize(x_0 = x0, T = T, return_trace=True)
                        o_time = time.time() - o_time
                        print("[l = {}] i: {}\tmeth: {}\tP: {}\tgamma: {}\tfx: {}".format(l, i, str(g), str(P), gm, ris['fx'].item()))
                        fx = ris['fx'].item()
                        if fx != fx or fx > f0:
                            g_max = gm
                        elif fx < c_val:
                            g_max = gm
                            c_val = fx
                        else:
                            g_min = gm
                            
                        if fx == fx and (best_fx is None or fx < best_fx):
                            best_fx = fx
                            best_gm = gm
                            exec_time = o_time

                    fl.write("{},{},{},{},{},{},{}\n".format(str(g) + "_" + str(P) +'_{}'.format(h_con),c, d, T, best_gm, best_fx, o_time))