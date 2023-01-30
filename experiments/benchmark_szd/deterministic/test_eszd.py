import torch
from eszd import ESZD
from eszd.directions import RandomCoords, HouseholderReflection, HouseholderQRDirections, FastRandomCoords

import matplotlib.pyplot as plt

def target(x, z=None):
    return x[:, :1].square().sum(dim=1, keepdims=True)

d = 20000
l = 5
device = 'cuda'
P = HouseholderReflection(d=d, l=l, dtype=torch.float64, device=device)
Pc = RandomCoords(d=d, l=l, dtype=torch.float64, device=device)
Pfc = FastRandomCoords(d=d, l=l, dtype=torch.float64, device=device)

x = torch.ones((1, d), dtype=torch.float64, device=device) + 5

evals = [target(x).item()]
evals_2 = [target(x).item()]
evals_3 = [target(x).item()]

opt_1 = ESZD(target, 
           x.clone(), 
           P, 
           alpha=lambda t: 0.5,
           h = lambda t: 1e-7, 
           verbose=True)
opt_2 = ESZD(target, 
           x.clone(), 
           Pc, 
           alpha=lambda t: 0.5, 
           h = lambda t: 1e-7, verbose=True)
opt_3 = ESZD(target, 
           x.clone(), 
           Pfc, 
           alpha=lambda t: 0.5, 
           h = lambda t: 1e-7, verbose=True)

budget = 50000
while opt_1.f_evals < budget:
    x = opt_1.step()
    evals.append(target(x).item())
while opt_2.f_evals < budget:
    x = opt_2.step()
    evals_2.append(target(x).item())
while opt_3.f_evals < budget:
    x = opt_3.step()
    evals_3.append(target(x).item())



fig, ax = plt.subplots()

ax.plot(range(len(evals)), evals, '-', label='householder')
ax.plot(range(len(evals_2)), evals_2, '-', label='coords')
ax.plot(range(len(evals_3)), evals_3, '-', label='fast coords')
ax.set_yscale("log")
ax.legend()
fig.savefig("./test_int_dim.pdf", bbox_inches='tight')
plt.close(fig)