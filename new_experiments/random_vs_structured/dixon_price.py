import torch

import numpy as np
import sys

from eszd import ESZD
from eszd.directions import GaussianDirections, SphericalDirections, RadamacherDirections, RandomCoords, HouseholderReflection 

import matplotlib.pyplot as plt

sys.path.append("../")
from synthetic_utils import Sphere, SparseMatrix, Rosenbrock, DixonPrice

from time import time

def plot_results(fun_name, results, out):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    for (label, fvals, _, ctime) in results:
        ax1.plot(range(len(fvals)), fvals, '-', label=label, lw=3)
        ax2.plot(range(len(ctime)), ctime, '-', label=label, lw=3)
        ax3.plot(ctime, fvals, '-', label=label)
    ax1.legend(fontsize=16)
    ax1.set_xlabel("$k$", fontsize=18)
    ax2.set_xlabel("$k$", fontsize=18)
    ax1.set_ylabel("$f(x_k)$", fontsize=18)
    ax3.set_ylabel("$f(x_k)$", fontsize=18)
    ax3.set_xlabel("cumulative time $(s)$", fontsize=18)
    ax2.set_ylabel("cumulative time $(s)$", fontsize=18)
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax3.set_yscale("log")
    fig.tight_layout()
    fig.savefig(f"{out}.pdf")
    plt.close(fig)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_title(fun_name, fontsize=20)
    for (label, _, std, _) in results:
        ax.plot(range(len(std)), std, '--', label=label, lw=3)
    ax.legend(fontsize=16)
    ax.set_xlabel("$k$", fontsize=18)
    ax.set_ylabel("std $f(x_k)$", fontsize=18)
    #ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(f"{out}_std.pdf")
    plt.close(fig)




def run_experiment(opt, T = 100, reps=1):
    fvals = [] #[opt.target(opt.x).item()]
    ctimes = []
    for r in range(reps):
        fvals.append([opt.target(opt.x).item()])
        ctimes.append([0.0])
        for k in range(T):
            it_time = time()
            x = opt.step()
            it_time = time() - it_time
          #  print("[--] r: {}, x: {}, y: {}".format(r, x, opt.target(x)))
            ctimes[r].append(it_time)
            fvals[r].append(opt.target(x).item())
        opt.reset()
#        print(opt.x)
    mu_vals, std_vals = np.mean(fvals, axis=0), np.std(fvals, axis=0)
    return mu_vals, std_vals, np.cumsum(ctimes, axis=1).mean(axis=0)      
        
        


d = 5#100
l = 5
dtype = torch.float64
device='cuda'

fun_1 = DixonPrice(d)
x0 = torch.full((1, d), 2.0, dtype=dtype, device=device)

h = lambda k: 1e-8#/np.sqrt(k)

alpha = lambda k : 6e-3#/np.sqrt(k)#.0
alpha_ref = lambda k : 4e-5#/np.sqrt(k)#.0
alpha_sp = lambda _ : 3e-4
alpha_g = lambda _ : 3e-6
alpha_rad = lambda _ : 5e-6


P1 = GaussianDirections(d=d, l=l, dtype=dtype, device=device)
P2 = SphericalDirections(d=d, l=l, dtype=dtype, device=device)
P3 = RandomCoords(d=d, l=l, dtype=dtype, device=device)
P4 = HouseholderReflection(d=d, l=l, dtype=dtype, device=device)
P5 = RadamacherDirections(d=d, l=l, dtype=dtype, device=device)

time()
P1()
P2()
P3()
P4()

opt_1 = ESZD(target = fun_1, x = x0.clone(), alpha=alpha_g, h=h, P=P1)
opt_2 = ESZD(target = fun_1, x = x0.clone(), alpha=alpha_sp, h=h, P=P2)
opt_3 = ESZD(target = fun_1, x = x0.clone(), alpha=alpha, h=h, P=P3)
opt_4 = ESZD(target = fun_1, x = x0.clone(), alpha=alpha_ref, h=h, P=P4)
opt_5 = ESZD(target = fun_1, x = x0.clone(), alpha=alpha_rad, h=h, P=P5)

T = 700

fvals_1, std_1, ctimes_1 = run_experiment(opt_1, T = T, reps=10)
fvals_2, std_2, ctimes_2 = run_experiment(opt_2, T = T, reps=10)
fvals_3, std_3, ctimes_3 = run_experiment(opt_3, T = T, reps=10)
fvals_4, std_4, ctimes_4 = run_experiment(opt_4, T = T, reps=10)
fvals_5, std_5, ctimes_5 = run_experiment(opt_5, T = T, reps=10)

results = [
    ('Random Gaussian',    fvals_1, std_1, ctimes_1),
    ('Random spherical',   fvals_2, std_2, ctimes_2),
    ('Random coordinates', fvals_3, std_3, ctimes_3),
    ('Random reflector',   fvals_4, std_4, ctimes_4),
    ('Random Rademacher',  fvals_5, std_5, ctimes_5),
]


plot_results("DixonPrice {}".format(d), results, "./dix_{}_{}".format(device, d))


