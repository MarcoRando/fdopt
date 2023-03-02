import torch

import numpy as np
import sys

from eszd import ESZD
from eszd.directions import GaussianDirections, SphericalDirections, RadamacherDirections, RandomCoords, HouseholderReflection 

import matplotlib.pyplot as plt

sys.path.append("../")
from synthetic_utils import Sphere, SparseMatrix

from time import time

def plot_results(results, out):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    for (label, fvals, ctime) in results:
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
    



def run_experiment(opt, T = 100):
    fvals = [opt.target(opt.x).item()]
    ctimes = [0.0]
    for k in range(T):
        it_time = time()
        x = opt.step()
        it_time = time() - it_time
        ctimes.append(it_time)
        fvals.append(opt.target(x).item())
    return fvals, np.cumsum(ctimes)        
        
        


d = 100
l = d // 10
dtype = torch.float64
device='cuda'

fun_1 = SparseMatrix(d, 4, dtype=dtype, device=device)
x0 = torch.full((1, d), 100.0, dtype=dtype, device=device)

alpha = lambda _ : 0.01
h = lambda k: 1e-7/k

alpha_g = lambda _ : 1e-5


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
opt_2 = ESZD(target = fun_1, x = x0.clone(), alpha=lambda _:5e-4, h=h, P=P2)
opt_3 = ESZD(target = fun_1, x = x0.clone(), alpha=alpha, h=h, P=P3)
opt_4 = ESZD(target = fun_1, x = x0.clone(), alpha=alpha, h=h, P=P4)
opt_5 = ESZD(target = fun_1, x = x0.clone(), alpha=alpha_g, h=h, P=P5)

T = 1000

fvals_1, ctimes_1 = run_experiment(opt_1, T = T)
fvals_2, ctimes_2 = run_experiment(opt_2, T = T)
fvals_3, ctimes_3 = run_experiment(opt_3, T = T)
fvals_4, ctimes_4 = run_experiment(opt_4, T = T)
fvals_5, ctimes_5 = run_experiment(opt_5, T = T)

results = [
    ('Random Gaussian', fvals_1, ctimes_1),
    ('Random spherical', fvals_2, ctimes_2),
    ('Random coordinates', fvals_3, ctimes_3),
    ('Random reflector', fvals_4, ctimes_4),
    ('Random Rademacher', fvals_5, ctimes_5),
]


plot_results(results, "./spmat_comp_{}".format(device))


