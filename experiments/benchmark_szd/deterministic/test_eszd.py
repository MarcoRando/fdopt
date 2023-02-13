import torch
import time
import numpy as np
from eszd import ESZD
from eszd.directions import SphericalDirections, HBDirections, RandomCoords, HouseholderReflection, HouseholderQRDirections, FastRandomCoords, HaarButterflyDirections

import matplotlib.pyplot as plt
import psutil

dint = 'a'

def target(x, z=None):
    if dint == 'a':
        return x.square().sum(dim=1, keepdims=True)
    return x[:, 10:100].square().sum(dim=1, keepdims=True)


def eval_optimizer(opt, evals, budget):
    ctimes = []
    while opt.f_evals < budget:
        it_time = time.time()
        x = opt.step()
        it_time = time.time() - it_time
        print("[--] f: {}/{} => {}s".format(opt.f_evals, budget, it_time))
        evals += [target(x).item() for _ in range(opt.P.l)]#.append(target(x).item())
        ctimes.append(it_time)
    return evals, np.cumsum(ctimes)

d = 1000
l = 100
device = 'cpu'

Ps = [
    (RandomCoords(d=d, l=l, dtype=torch.float64, device=device), 'coordinates'), # Random coords
    (HouseholderReflection(d=d, l=l, seed=13, dtype=torch.float64, device=device), 'random reflector'), # Random coords
    (HouseholderReflection(d=d, l=l, permutation=True, dtype=torch.float64, device=device), 'random permuted reflector'), # Random coords
#    (HouseholderReflection(d=d, l=l, seed=13, dtype=torch.float64, num_reflection=2, device=device), 'random reflectors [2]'), # Random coords
    (SphericalDirections(d=d, l=l, dtype=torch.float64, device=device), 'spherical'),
    (HouseholderQRDirections(d=d, l=l, dtype=torch.float64, device=device), 'QR'),
#    (HaarButterflyDirections(d=d, l=l, dtype=torch.float64, dct=False, device=device), 'Haar Butterfly'),
  #  (HaarButterflyDirections(d=d, l=l, dtype=torch.float64, permute=True, device=device), 'Permuted Haar Butterfly'),
#    (FastRandomCoords(d=d, l=l, dtype=torch.float64, device=device), 'fast coordinates'), # Fast coords
 #   (RandomCoords(d=d, l=l, dct=True, dtype=torch.float64, device=device), 'dct'), # DCT coords
 #   (FastRandomCoords(d=d, l=l, dct=True, dtype=torch.float64, device=device), 'fixed dct'),# Fast transformed coords
 #   (HBDirections(d=d, l=l, dct=False, dtype=torch.float64, device=device), 'extended butterfly'),
 #   (HBDirections(d=d, l=l, dct=True, dtype=torch.float64, device=device), 'extended dct butterfly'),
 #   (HBDirections(d=d, l=l, dct=True, permute=True, dtype=torch.float64, device=device), 'ep dct butterfly'),
]


x = torch.full((1, d),100, dtype=torch.float64, device=device) + 20 * torch.randn(size=(1, d), dtype=torch.float64, device=device)# + 100

#evals = [target(x).item()]
#evals_2 = [target(x).item()]
#evals_3 = [target(x).item()]
#evals_4 = [target(x).item()]

def build_opt(target, x, P, a=0.5):
    P()
    return ESZD(target, x.clone(), P, alpha=lambda t: a, h = lambda t: 1e-5/t, verbose=True)
opts = []
alps = [0.1, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
labels = []
for i, (P, label) in enumerate(Ps):
    opts.append(build_opt(target, x, P, alps[i]))
    labels.append(label)
budget = 50000
#def eval_optimizer(opt, evals, budget):
evals = []
times = []
for i in range(len(opts)):
    values, ctime = eval_optimizer(opts[i], [target(x).item()], budget)
    evals.append(values)
    times.append(ctime)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for i in range(len(opts)):
    ax1.plot(range(len(evals[i])), evals[i], '-', label=labels[i], lw=3)
    ax2.plot(range(len(times[i])), times[i], '-', label=labels[i], lw=3)
ax1.set_yscale("log")
ax1.legend()
ax1.set_xlabel("function evaluations")
ax1.set_ylabel("$f(x_k)$")
ax2.set_yscale("log")
ax2.set_xlabel("iterations")
ax2.set_ylabel("$s$")
ax2.legend()
fig.tight_layout()
fig.savefig("./test_{}_{}_{}.pdf".format(device, len(Ps)-1, dint), bbox_inches='tight')
plt.close(fig)