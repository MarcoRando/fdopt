import torch
import time
import sys

from fdopt.grad_estimators import EfficientGSGA, ForwardFD, GenStructGradApprox as GSGA
from fdopt.direction_generators import SphericalDirections, QRDirections,GaussianDirections, CoordinateDirections
from fdopt.stepsize import ConstantStepsize

sys.path.append("../targets")
from synthetic import SphereFunction

d = 1000
l = 900

f = SphereFunction(d =d)

h = ConstantStepsize(1e-5)
P = QRDirections(d=l, l=l)
g = EfficientGSGA(fixed_matrix_mode='identity', eps=1e-1, P = P, d = d, l=l, f = f)
g2 = ForwardFD(d =d, l=l, f=f, P = CoordinateDirections(d =d, l = l))
g3 = GSGA(P = P, d =d , l= l, f=f)

#x = torch.randn((1, d), dtype=torch.float64)
x = torch.ones((1, d), dtype=torch.float64) + 123 + torch.rand(1)

fx = f(x)
g1 = g(x, fx,  h=1e-5)

egsga_time = time.time()
g1 = g(x, fx,  h=1e-5)
egsga_time = time.time() - egsga_time

ffd_time = time.time()
g2 = g2(x, fx,  h=1e-5)
ffd_time = time.time() - ffd_time

gsga_time = time.time()
g3 = g3(x, fx,  h=1e-5)
gsga_time = time.time() - gsga_time

real_grad = f.grad(x)

print("EGSG-QR", torch.linalg.norm(g1 - real_grad)/torch.linalg.norm(real_grad))
print("FFD-QR", torch.linalg.norm(g2 - real_grad)/torch.linalg.norm(real_grad))
print("GSGA-QR", torch.linalg.norm(g3 - real_grad)/torch.linalg.norm(real_grad))

print("EGSG-QR time", egsga_time)
print("FFD-QR time", ffd_time)
print("GSGA-QR time", gsga_time)


#print(g, f.grad(x), g2)