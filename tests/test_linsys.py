import torch

from torch_dct import dct, idct

dtype = torch.float64

d = 4
l = 1


def target(x):
#    x = x.reshape(1, -1)
    return 0.5 * x.square().sum(dim = 1, keepdim=True)

def grad(x):
    return x.reshape(d, -1)


A = torch.randn(l, l, dtype=dtype)
Q, R = torch.linalg.qr(A)


E = torch.eye(l, d, dtype=dtype)

E = dct(E, norm='ortho')
x = torch.rand(1, d, dtype=dtype) + 12 #torch.full((1, d), 12.0, dtype = dtype)
h = 1e-10
V = Q @ E
print(V + x)
fx = target(x)
vals = (target(x + h * V) - fx)/h
#alpha = [target()]
print(() @ vals, grad(x))
#print(E.T)

#print(torch.linalg.pinv(E))
#print(Q.T)
#print(E.T @ Q.T)


#V = Q @ E
#
#
#
#
##print(V)
#x = torch.rand(1, d, dtype=dtype) + 12 #torch.full((1, d), 12.0, dtype = dtype)
##print(target(x))
#fx = target(x)
#h = 1e-9 * l/d
#D = torch.tensor([(target(x + h * V[i]) - fx)/h for i in range(l)], dtype = dtype)
#
#g_a = torch.zeros((d, 1))
#for i in range(l):
##    print(((target(x + h * V[i]) - fx)/h)*V[i])
#    v = ((target(x + h * V[i]) - fx)/h)*V[i]
#    g_a += v.reshape(d, 1) #((target(x + h * V[i].reshape(x.shape)).item() - fx.item())/h)*V[i]
#g_a = (d/l) * g_a #torch.tensor([((target(x + h * V[i]).item() - fx.item())/h)*V[i] for i in range(l)], dtype=dtype).reshape(-1, d).sum(dim=0)
##g =  torch.linalg.lstsq(torch.eye(l, d, dtype=dtype), torch.linalg.inv(Q) @ D)[0]
##print(D, grad(x), g)
#g = dct(torch.linalg.pinv(V)) @ D
##print(D, g, V)
##print(dct(g))
#A = dct(V, norm='ortho')
#
#g = torch.linalg.pinv(A) @ D
#
#
#p = (torch.linalg.pinv(E))#.to_sparse()
#print(p)
#print(p == E.T)
#print(p.to_dense())
#g = (E.T @ Q).mv(D)
#print(g)
#