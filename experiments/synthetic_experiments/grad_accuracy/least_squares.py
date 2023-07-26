import torch
import sys
import matplotlib.pyplot as plt
sys.path.append("../../targets/")
from synthetic import LeastSquares

from fdopt import ZD
from fdopt.grad_estimators import ForwardFD
from fdopt.stepsize import Stepsize, ConstantStepsize
from fdopt.direction_generators import CoordinateDirections

d = 4
l = 1


x_star = torch.ones((1, d), dtype=torch.float64) + 1
f = LeastSquares(m =d, d= d, x_star=x_star, dtype=torch.float64)

x0 = torch.randn((1, d), dtype=torch.float64)
P = CoordinateDirections(d =d , l = l, dtype = torch.float64)
g = ForwardFD(d = d, l = l, f = f, P = P)

gamma = ConstantStepsize(0.1)#Stepsize(gamma_0=0.5, law=lambda t : 1/(t + 1))
h = ConstantStepsize(1e-5)

opt = ZD(g = g, gamma = gamma, h = h)


ris = opt.optimize(x0, T = 100, return_trace=True)

print(ris['x'], ris['fx'])

rr = []
for (x, y) in ris['trace']:
    print(x)
    rr.append(y.item())
    
fig, ax = plt.subplots()
ax.plot(rr, '-')
ax.set_yscale("log")
fig.savefig("./test.pdf")
