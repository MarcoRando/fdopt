import torch
import sys
from fdopt import ZD
from fdopt.grad_estimators import ForwardFD, CentralFD, LinInterpolation, CentralLinInterpolation
from fdopt.direction_generators import GaussianDirections, RademacherDirections, SphericalDirections, CoordinateDirections, QRDirections

from fdopt.stepsize import ConstantStepsize

sys.path.append("../../targets/")
from synthetic import LeastSquares

m = d = 2
L = 200
mu = 50
device = 'cuda'
dtype = torch.float64
target = LeastSquares(m = m, d=d, device=device, dtype=dtype, L = L, mu =mu)

grad = target.grad

def compute_grad_accuracy(target, methods):



x = torch.ones((1, d), device=device, dtype=dtype)

P = SphericalDirections(d = d, l = d, device=device, dtype=dtype)
g_est = ForwardFD(target, P = P, normalize=True)

h = 1e-10
fx = target(x)
print(g_est(x, fx, h))
print(grad(x))
print(torch.linalg.norm(g_est(x, fx, h) - grad(x)).square())



