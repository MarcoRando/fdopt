import numpy as np

from fdopt import ZD
from fdopt.stepsize import ConstantStepsize
from fdopt.direction_generators import CoordinateDirections

def target(x):    
    return np.sum(x**2)

d, l = 12, 2
T = 10

h = ConstantStepsize(gamma_0 = 1e-10)
P = CoordinateDirections(d = d, l = l)
gamma = ConstantStepsize(gamma_0 = 0.1 * l/d)

opt = ZD(gamma=gamma, h=h, P=P, g_estimator='central', parallel_execution=('thread',4))


x0 = np.asarray([i for i in range(d)])
result = opt.optimize(target, x_0=x0, T=T, return_trace=True)

print("[--] x = {}\tf(x) = {}".format(result['x'], result['fx']))


