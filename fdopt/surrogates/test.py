import torch
import time
from surrogates import ForwardFD, CentralFD, OptimizedForwardFD

from fdopt.direction_generators import Householder, SphericalDirections

def f(x):
    print(x.shape)
    return x.square().sum(axis=1)

def grad_f(x):
    return 2 * x

d = 5
P = Householder(d = d, l = d)

g = OptimizedForwardFD() #OptCentralFD(vectorized=True)

x0 = torch.tensor([i for i in range(d)], dtype=torch.float64).reshape(1, -1)
g_time = time.time()
g_k = g(f, x0, h = 1e-5, P=P())
g_time = time.time() - g_time
#0.03985786437988281 seconds
#0.014542579650878906 seconds
print(g_k, g_time)
print(torch.linalg.norm(g_k - grad_f(x0)).square() / torch.linalg.norm(grad_f(x0)).square() )
