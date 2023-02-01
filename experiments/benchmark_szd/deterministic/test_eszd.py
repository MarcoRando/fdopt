import torch
import time
import numpy as np
from eszd import ESZD
from eszd.directions import RandomCoords, HouseholderReflection, HouseholderQRDirections, FastRandomCoords

import matplotlib.pyplot as plt

def target(x, z=None):
    return x[:, 10:100].square().sum(dim=1, keepdims=True)


def eval_optimizer(opt, evals, budget):
    ctimes = []
    while opt.f_evals < budget:
        it_time = time.time()
        x = opt.step()
        it_time = time.time() - it_time
        evals.append(target(x).item())
        ctimes.append(it_time)
    return evals, np.cumsum(ctimes)

d = 10000
l = 1000
device = 'cuda'

Ps = [
    (RandomCoords(d=d, l=l, dtype=torch.float64, device=device), 'coordinates'), # Random coords
    (HouseholderReflection(d=d, l=l, dtype=torch.float64, device=device), 'random reflector'), # Random coords
    (HouseholderQRDirections(d=d, l=l, dtype=torch.float64, device=device), 'QR'),
    (FastRandomCoords(d=d, l=l, dtype=torch.float64, device=device), 'fast coordinates'), # Fast coords
    (RandomCoords(d=d, l=l, dct=True, dtype=torch.float64, device=device), 'dct'), # DCT coords
    (FastRandomCoords(d=d, l=l, dct=True, dtype=torch.float64, device=device), 'fast dct') # Fast transformed coords
]


x = torch.full((1, d),1000, dtype=torch.float64, device=device) + 20 * torch.randn(size=(1, d), dtype=torch.float64, device=device)# + 100

#evals = [target(x).item()]
#evals_2 = [target(x).item()]
#evals_3 = [target(x).item()]
#evals_4 = [target(x).item()]

def build_opt(target, x, P, a=0.5):
    P()
    return ESZD(target, x.clone(), P, alpha=lambda t: a, h = lambda t: 1e-7, verbose=True)
opts = []
alps = [1.0, 1.0, 5.0, 1.0, 5.0, 5.0]
labels = []
for i, (P, label) in enumerate(Ps):
    opts.append(build_opt(target, x, P, alps[i]))
    labels.append(label)
budget = 50000#50000
#def eval_optimizer(opt, evals, budget):
evals = []
times = []
for i in range(len(opts)):
    values, ctime = eval_optimizer(opts[i], [target(x).item()], budget)
    evals.append(values)
    times.append(ctime)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for i in range(len(opts)):
    ax1.plot(range(len(evals[i])), evals[i], '-', label=labels[i])
    ax2.plot(range(len(times[i])), times[i], '-', label=labels[i])
ax1.set_yscale("log")
ax1.legend()
ax2.set_yscale("log")
ax2.legend()
fig.tight_layout()
fig.savefig("./test_{}.pdf".format(device), bbox_inches='tight')
plt.close(fig)