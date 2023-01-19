import torch
import os
from benchmark_functions import SphereFunction

from eszd import SSZD, SSVRZD, ESSZD
from eszd.target_functions import FiniteSumTarget
from eszd.directions import SphericalDirections, StructuredSphericalDirections, GaussianDirections
import numpy as np
import matplotlib.pyplot as plt
from utils import *

class LinearRegression(FiniteSumTarget):
    #    def __init__(self, name, x_min, f_min, seed, num_elem) -> None:
    def __init__(self, d, num_elems, seed, device = 'cpu', dtype=torch.float32) -> None:
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)
        self.w = torch.rand(size=(1, d), generator = self.generator, device=device, dtype=dtype)
        self.X = torch.rand((num_elems, d), generator=self.generator, device=device, dtype=dtype) * 10 + 5
        self.y = torch.matmul(self.w,self.X.T).reshape(-1)
        super().__init__("Linear Regression", self.w, 0.0, seed, num_elems)

    def __call__(self, w_hat, idx = None):
        if idx is None:
            return (torch.matmul(w_hat , self.X.T) - self.y).square().mean(axis=1)
#        print((torch.matmul(w_hat , self.X[idx, :]) - self.y[idx]).square())
        return (torch.matmul(w_hat , self.X[idx, :]) - self.y[idx]).square()


os.makedirs("./results/comparisons", exist_ok=True)

d = 8000# int(1e4)

m = 10
l = 100

budget = (d + m * (2 * l + 2)) * 100

print(budget)
print(budget / l)
print(budget / (d + m * (2 * l + 2)))



dtype = torch.float64
device= 'cuda'
target = LinearRegression(d=d, num_elems=100, seed=1212, device=device, dtype=dtype)#SphereFunction(d)
#    def __init__(self, target, x0, P : DirectionMatrix, alpha, h):
L = torch.linalg.eigvalsh(target.X.T @ target.X).max()
x = torch.zeros(1, d, dtype=dtype, device=device)

print("[--] L: {}".format(L))
#exit()
#print(target(target.w))#, target.w)
alpha = lambda t : l/(d * L * sqrt(t)) #if 1/sqrt(t) > 1e-7 else 1e-7
alpha_vr = lambda t : l/(d * L) #if 1/sqrt(t) > 1e-7 else 1e-7

h = lambda t : 1e-5

sampler = torch.Generator()
sampler.manual_seed(121314)


def sampling_procedure():
    return torch.randint(low=0, high=target.X.shape[0], size=(1,), generator=sampler).item()

def get_results(opts):
    results = {}
    for (name, opt, params) in opts:
        results[name] = opt.optimize(**params) 
    return results

def plot_time(d, l, m, results, out):
    fig, ax = plt.subplots()
    ax.set_title("Cumulative time")
    for (name, ris) in results.items():
        vals = [0]
        times = np.cumsum(ris[1])
        if "SSVRZD" in name:
            for x in times:
                vals += [x for _ in range(d + m * (2*l + 2))]
        elif "SSZD" in name:
            for x in times:
                vals += [x for _ in range(l)]
        ax.plot(range(len(vals)), vals, '-', label=name)
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("function evaluations")
    ax.set_ylabel("$s$")
    plt.savefig("./{}_cumulative_time.png".format(out), bbox_inches='tight')
    plt.close(fig)
    
def plot_results(d, l, m, results, out):
    fig, ax = plt.subplots()
    ax.set_title("Function Values")
    for (name, ris) in results.items():
        vals = []
        if "SSZD" in name or "ESSZD" in name:
            print("x> ",name)
            for i, x in enumerate(ris[2]):
                x = x.cpu().numpy()
                vals += [x for _ in range(ris[3][i])]
        elif "SSVRZD" in name:
            print("=> ",name)
            for i, x in enumerate(ris[2]):
                x = x.cpu().numpy()
                vals += [x for _ in range(ris[3][i])]
        print("{} => {}".format(name, len(vals)))
        ax.plot(range(len(vals)), vals, '-', label=name)
    ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("function evaluations")
    ax.set_ylabel("$f(x)$")
    plt.savefig("./{}_funvalues.png".format(out), bbox_inches='tight')
    plt.close(fig)
    plot_time(d, l, m, results, out)

sszd_ref = SSZD(ReflectedDirections(d=d, l=l, dtype=dtype, device=device,), alpha, h, sampling_procedure=sampling_procedure)
#sszd_but = SSZD(ButterflyDirections(d=d, l=l, dtype=dtype, device=device,), alpha, h, sampling_procedure=sampling_procedure)
sszd_coo = SSZD(RandomCoords(d=d, l=l, dtype=dtype, device=device,), alpha_vr, h, sampling_procedure=sampling_procedure)
sszd_coo_2 = SSZD(RandomCoords(d=d, l=l, dtype=dtype, device=device,), alpha, h, sampling_procedure=sampling_procedure)


sszd_perm = SSZD(PermutedDirections(d=d, l=l, dtype=dtype, device=device,), alpha, h, sampling_procedure=sampling_procedure)

ssvrzd_coo = SSVRZD(RandomCoords(d=d, l=l, dtype=dtype, device=device,), alpha_vr, h, sampling_procedure=sampling_procedure, update_P=1)
ssvrzd_ref = SSVRZD(ReflectedDirections(d=d, l=l, dtype=dtype, device=device,), alpha_vr, h, sampling_procedure=sampling_procedure, update_P=1)
ssvrzd_per = SSVRZD(PermutedDirections(d=d, l=l, dtype=dtype, device=device,), alpha, h, sampling_procedure=sampling_procedure, update_P=1)

#P : DirectionMatrix, Lk, h, theta = 1.0, sampling_procedure = None)
esszd_coo = ESSZD(RandomCoords(d=d, l=l, dtype=dtype, device=device,), 1e5, h, theta=0.001, sampling_procedure=sampling_procedure)
esszd_ref = ESSZD(ReflectedDirections(d=d, l=l, dtype=dtype, device=device,), 1e5, h, theta=0.001, sampling_procedure=sampling_procedure)

#
##    def optimize(self, target, x, T = 1000, eps = 1e-5):
#T = 20




print(l, "=>", budget/l)

print((d + m * (2 * l + 2)), '=>', budget / (d + m * (2 * l + 2)))
#exit()

szd_params = {
    'target' : target,
    'x' : x,
    'T' : budget // l,
}

vrzd_params = {
    'target' : target,
    'x' : x,
    'm' : m,
    'T' : budget // (d + m * (2 * l + 2)),
}


results = get_results([
    ("ESSZD [Coordinates]", esszd_coo, szd_params),
    ("ESSZD [Reflected]", esszd_ref, szd_params),
    ("SSVRZD [Reflected]", ssvrzd_ref, vrzd_params),
#    ("SSVRZD [Permuted]", ssvrzd_ref, vrzd_params),
#
#    ("SSZD [Reflected]", sszd_ref, szd_params),
#    ("SSZD [Coordinates] const step", sszd_coo, szd_params),
    ("SSZD [Coordinates]", sszd_coo_2, szd_params),
    ("SSVRZD [Coordinates]", ssvrzd_coo, vrzd_params),
#    ("SSZD [Permuted]", sszd_perm, szd_params),
    ])

plot_results(d, l, m, results, "linear_reg")
#x, mean_g_time, trace = sszd.optimize(target, x, T=int(1e6), eps=1e-5)
#print(np.mean(mean_g_time))
#print(x)


