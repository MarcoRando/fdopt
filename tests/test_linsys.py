import torch

def target(x):
    x = x.reshape(1, -1)
    return 0.5 * x.square().sum(dim = 1)

def grad(x):
    return x

dtype = torch.float64

d = 10
l = 9

A = torch.randn(d, d, dtype=dtype)
Q, _ = torch.linalg.qr(A)

V = torch.eye(l, d, dtype=dtype) @ Q

print(V)
x = torch.rand(1, d, dtype=dtype) #torch.full((1, d), 12.0, dtype = dtype)
print(target(x))
fx = target(x)
h = 1e-9 * l/d
D = torch.tensor([(target(x + h * V[i]) - fx)/h for i in range(l)], dtype = dtype)

g_a = torch.zeros((d, 1))
for i in range(l):
    print(((target(x + h * V[i]) - fx)/h)*V[i])
    v = ((target(x + h * V[i]) - fx)/h)*V[i]
    g_a += v.reshape(d, 1) #((target(x + h * V[i].reshape(x.shape)).item() - fx.item())/h)*V[i]
g_a = (d/l) * g_a #torch.tensor([((target(x + h * V[i]).item() - fx.item())/h)*V[i] for i in range(l)], dtype=dtype).reshape(-1, d).sum(dim=0)
g =  torch.linalg.lstsq(V, D)[0]
#print(D, grad(x), g)

print(torch.linalg.norm(g_a.reshape(1, -1) - g))
print("g_a", g_a)
print("g", g)
print(torch.linalg.norm(grad(x) -g))