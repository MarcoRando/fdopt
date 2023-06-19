import torch

from torch_dct import dct, idct
import sys

sys.path.append("../experiments/targets/")

from synthetic import LeastSquares

dtype = torch.float64

d = 10


target = LeastSquares(m = d, d = d, seed=121212, dtype=dtype, L = 100, mu=1)

def target(x):
#    x = x.reshape(1, -1)
    return 0.5 * x.square().sum(dim = 1, keepdim=True)

#grad = target.grad

def grad(x):
    return x.reshape(d, -1)
l = 2
gen = torch.Generator()
A = torch.randn(l, l, dtype=dtype, generator=gen)
Q, R = torch.linalg.qr(A)

#Q = Q.T

E = torch.eye(l, d, dtype=dtype)

E = dct(E, norm='ortho')
Epinv = torch.linalg.pinv(E)
x = torch.ones(1, d, dtype=dtype) + 12 #torch.full((1, d), 12.0, dtype = dtype)
h = 1e-7
V = Q @ E
#print(V @ V.T)
#print(V + x)
fx = target(x)
vals = (target(x + h * V) - fx)/h
#print(vals)
#vals = vals.T
#    print(l, vals)

g1 = Epinv @ Q.T @ vals
B = torch.linalg.qr(torch.randn(l, l, dtype=dtype, generator=gen))[0]
V2 = B @ E

x = torch.ones(1, d, dtype=dtype)
fx = target(x)
vals2 = (target(x + h * V2) - fx)/h
g2 = Epinv @ B.T @ vals2
#print(Q.T, B.T)
#print(vals, vals2)
#print(x)
print(g1.reshape(-1))
print(g2.reshape(-1))
print(grad(x).reshape(-1))

# #print(vals)
# #alpha = [target()]
# g = Epinv @  Q.T @ vals 
g2 = torch.linalg.lstsq(V.T @ V, V.T @vals)[0]
g3 = torch.linalg.lstsq(V2.T @ V2, V2.T @ vals2)[0]
print(g2.reshape(-1))
print(g3.reshape(-1))

# gx = grad(x)

# #    print('g', g.reshape(-1))
# #    print('grad', gx.reshape(-1))
# #    print(torch.linalg.norm(g.reshape(-1)-gx.reshape(-1)))
# #print("l", l)
# #print(g.reshape(-1), '\n', gx.reshape(-1))
# #print('lxl rotation', (torch.linalg.norm(g.reshape(-1)-gx.reshape(-1), ord=2)/torch.linalg.norm(gx.reshape(-1), ord=2).item()))
# #print('lxl rotation [g2]', (torch.linalg.norm(g2.reshape(-1)-gx.reshape(-1), ord=2)/torch.linalg.norm(gx.reshape(-1), ord=2).item()))

# A = torch.randn(d, d, dtype=dtype)
# Q, R = torch.linalg.qr(A)
# V = torch.eye(l, d, dtype=dtype) @ Q
# vals = (target(x + h * V) - fx)
# g = torch.linalg.lstsq(h * V.T @ V, V.T @vals)[0]
# print('dxd rotation [lstsq]', (torch.linalg.norm(g-grad(x))/ torch.linalg.norm(grad(x))).item())
# gb = (1/h) * torch.linalg.pinv(V) @ vals
# print('dxd rotation [pinv]',(torch.linalg.norm(gb-grad(x))/ torch.linalg.norm(grad(x))).item())

# #g2 = torch.linalg.solve(h * V, vals)
# #print('scheinberg', (torch.linalg.norm(g2-grad(x))/ torch.linalg.norm(grad(x))).item())





# #print(E.T)

# #print(torch.linalg.pinv(E))
# #print(Q.T)
# #print(E.T @ Q.T)


# #V = Q @ E
# #
# #
# #
# #
# ##print(V)
# #x = torch.rand(1, d, dtype=dtype) + 12 #torch.full((1, d), 12.0, dtype = dtype)
# ##print(target(x))
# #fx = target(x)
# #h = 1e-9 * l/d
# #D = torch.tensor([(target(x + h * V[i]) - fx)/h for i in range(l)], dtype = dtype)
# #
# #g_a = torch.zeros((d, 1))
# #for i in range(l):
# ##    print(((target(x + h * V[i]) - fx)/h)*V[i])
# #    v = ((target(x + h * V[i]) - fx)/h)*V[i]
# #    g_a += v.reshape(d, 1) #((target(x + h * V[i].reshape(x.shape)).item() - fx.item())/h)*V[i]
# #g_a = (d/l) * g_a #torch.tensor([((target(x + h * V[i]).item() - fx.item())/h)*V[i] for i in range(l)], dtype=dtype).reshape(-1, d).sum(dim=0)
# ##g =  torch.linalg.lstsq(torch.eye(l, d, dtype=dtype), torch.linalg.inv(Q) @ D)[0]
# ##print(D, grad(x), g)
# #g = dct(torch.linalg.pinv(V)) @ D
# ##print(D, g, V)
# ##print(dct(g))
# #A = dct(V, norm='ortho')
# #
# #g = torch.linalg.pinv(A) @ D
# #
# #
# #p = (torch.linalg.pinv(E))#.to_sparse()
# #print(p)
# #print(p == E.T)
# #print(p.to_dense())
# #g = (E.T @ Q).mv(D)
# #print(g)
# #