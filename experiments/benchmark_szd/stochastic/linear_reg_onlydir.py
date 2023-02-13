import os
import sys
import torch
import time
import numpy as np
from eszd import ESZD, ESVRZD
from eszd.directions import HouseholderReflection, RandomCoords
import pyproximal as pyprox


from utils import LinearRegr, SquareFun, LinearRegrNumpy
import matplotlib.pyplot as plt

from math import sqrt

sys.path.append("../ZORO/")
sys.path.append("../ZO-BCD/pyZO-BCD")
from optimizers import ZORO, AdaZORO
from bcd_optimizers import ZOBCD


def plot_results(labels, fvals, ctimes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.set_xlabel("function evaluations")
    ax1.set_ylabel("$f(x_k)$")
    ax1.set_yscale('log')
    for i in range(len(labels)):
        ax1.plot(range(len(fvals[i])), fvals[i], '-', label=labels[i], lw=3)
    ax1.legend()
    ax2.set_xlabel("function evaluations")
    ax2.set_ylabel("$s$")
    ax2.set_yscale("log")
    for i in range(len(labels)):
        ax2.plot(range(len(ctimes[i])), ctimes[i], '-', label=labels[i], lw=3)
    fig.tight_layout()
    fig.savefig("{}/comparison.pdf".format(out), bbox_inches='tight')
    plt.close(fig)

d = 1000
l = d//3
num_points = 750
dtype = torch.float64
out = "./results/only_P"
os.makedirs(out, exist_ok=True)
target = LinearRegr(num_points=num_points, d =d ,dtype=dtype, seed=12)
target_zoro = LinearRegrNumpy(num_points=num_points, d=d, dtype=dtype,seed=12)


x0 = torch.ones( d, dtype=dtype)
P = HouseholderReflection(d=d, l = l, permutation=True, dtype=dtype,seed=12)
#P = RandomCoords(d=d, l=l, dct=True)
alpha = lambda k :  1e-4 * 1/sqrt(k)

h = lambda k : 1e-9 #/sqrt(k)

def sampling():
    return torch.randint(0, num_points, size=(1, 1))[0, 0]

L = torch.linalg.eigvalsh(target.X.T @ target.X).max()
def test_adazoro(budget, prox=None):
    noiseamp = 0.00 # noise amplitude

    # Choose initialization
    xx0   = x0.clone().numpy()

#    sparsity = s
    init_sparsity = int(0.1*len(xx0))#2   # initial sparsity guess for adaptive sampling, i.e. AdaZORO

    # Parameters for ZORO. Defaults are fine in most cases
    params = {"step_size":lambda t : 1e-4 * 1/sqrt(t), "delta": 1e-7, "max_cosamp_iter": 20, 
            "cosamp_tol": 0.5,"sparsity": init_sparsity,
            "num_samples_constant": 1, "phi_cosamp": 0.4,"phi_lstsq": 0.05, 
            "compessible_constant": 1.05}

    performance_log_ZORO = [target_zoro(xx0)]
    ctimes = [0]

    # initialize optimizer object
    opt  = AdaZORO(xx0, target_zoro, params, sampler=sampling, function_budget= budget, prox=prox)
    evals_ZORO = 0
    while evals_ZORO < budget:
        it_time =time.time()
        x, evals_ZORO, solution_ZORO, termination, _ = opt.step()
        it_time = time.time() - it_time
        y = target_zoro(x)
        # save some useful values
        performance_log_ZORO +=  [y for _ in range(l)]
        ctimes.append(it_time)
        print("[--] y: {}".format(y))
    return np.cumsum(ctimes), performance_log_ZORO

def test_zoro(budget, prox = None):
    noiseamp = 0.00 # noise amplitude

    # Choose initialization
    xx0   = x0.clone().numpy()

#    sparsity = s
    sparsity = int(0.1*len(xx0)) # This is a decent default, if no better estimate is known. 

    # Parameters for ZORO. Defaults are fine in most cases
    params = {"step_size":alpha, "delta": 0.0001, "max_cosamp_iter": 10, 
            "cosamp_tol": 0.25,"sparsity": sparsity, 
            "num_samples": l}

    performance_log_ZORO = [target_zoro(xx0)]
    ctimes = [0]

    # initialize optimizer object
    opt  = ZORO(xx0, target_zoro, params, sampler=sampling, function_budget= budget, prox=prox)
    evals_ZORO = 0
    while evals_ZORO < budget:
        it_time =time.time()
        x, evals_ZORO, solution_ZORO, termination = opt.step()
        it_time = time.time() - it_time
        y = target_zoro(x)
        # save some useful values
        performance_log_ZORO +=  [y for _ in range(l)]
        ctimes += [it_time for _ in range(l)]
        print("[--] y: {}".format(y))
    return np.cumsum(ctimes), performance_log_ZORO



def test_zobcd(budget, type = 'ZOBCD-R'):
    noiseamp = 0.00 # noise amplitude

    # Choose initialization
    xx0   = x0.clone().numpy()

#    sparsity = s
    sparsity = int(0.1*len(xx0)) # This is a decent default, if no better estimate is known. 

    # Parameters for ZORO. Defaults are fine in most cases
    params =  {"Type": type, "sparsity": sparsity, "delta": 1e-7, "J": 20}

    performance_log_ZORO = [target_zoro(xx0)]
    ctimes = [0]

    # initialize optimizer object
    opt  = ZOBCD(xx0, alpha, target_zoro, params, sampler=sampling, function_budget= budget)
    evals_ZORO = 0
    while evals_ZORO < budget:
        it_time =time.time()
        x, evals_ZORO, solution_ZORO, termination, num_samples = opt.step()
        it_time = time.time() - it_time
        y = target_zoro(x)
        # save some useful values
        performance_log_ZORO +=  [y for _ in range(num_samples)]
        ctimes += [it_time for _ in range(num_samples)]
        print("[--] y: {}\t{}".format(y, evals_ZORO))
    return np.cumsum(ctimes), performance_log_ZORO


def test_eszd(budget):
    opt = ESZD(target, x0.clone(), P, alpha=alpha, h=h, sampling_fun=sampling)
    ctimes, fvals = [0], [target(x0)]
    while opt.f_evals < budget:
        it_time = time.time()
        x = opt.step()
        it_time = time.time() - it_time
        print("f(x) = {}".format(target(x)))
        ctimes +=[it_time for _ in range(l)]
        fvals += [target(x) for _ in range(l)] #.append(target(x))
    return np.cumsum(ctimes), fvals

def test_esvrzd(budget):
    m = 10
    opt = ESVRZD(target, x0.clone(), P, alpha=lambda k : 1e-4, h=h, m=m, sampling_fun=sampling)
    ctimes, fvals = [0], [target(x0)]
    while opt.f_evals < budget:
        it_time = time.time()
        x = opt.step()
        it_time = time.time() - it_time
        print("f(x) = {}".format(target(x)))
        ctimes +=[it_time for _ in range((2*l + 2)*m + (l + 1))]
        fvals += [target(x) for _ in range((2*l + 2)*m + (l + 1))] #.append(target(x))
#    print(opt.f_evals)
#    exit()
    return np.cumsum(ctimes), fvals


#print(1/L)

sigma = 0.1 # regularization weight
ell_1_prox = pyprox.L1(sigma=sigma)

budget = 140280


esvrzd_ris = test_esvrzd(budget)
zobcd_ris = test_zobcd(budget)
zobcdrc_ris = test_zobcd(budget, 'ZOBCD-RC')
eszd_ris = test_eszd(budget)
zoro_ris = test_zoro(budget)
prox_zoro_ris = test_zoro(budget, ell_1_prox)
#
#
ctimes = [
    zoro_ris[0],  
    prox_zoro_ris[0],  
    zobcd_ris[0], 
    zobcdrc_ris[0],  
    eszd_ris[0], 
 #   esvrzd_ris[0]
    ]
fvals =  [
    zoro_ris[1],  
    prox_zoro_ris[1],  
    zobcd_ris[1], 
    zobcdrc_ris[1],  
    eszd_ris[1], 
#    esvrzd_ris[1]
    ]
labels = [
    'ZORO',  
    'Prox-Zoro', 
    'ZO-BCD', 
    'ZO-BCD-RC', 
    'S-SZD', 
    #'S-SVRZD'
    ]

plot_results(labels, fvals, ctimes)



