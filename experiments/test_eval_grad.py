import torch

def target(x):
    return x.square().sum(dim=1, keepdims=False)

d = 4
l=2

P = torch.eye(l, d)

h = 0.5

x = torch.ones((1, d))
fx = target(x)

print(x + h*P)

for i in range(P.shape[0]):
    print(x + h * P[i, :])

g = P.T.mv(target(x + h*P).div_(h))


print(f"g = {g}")


d = l= 1000
from time import time

it_time = time()
P = torch.eye(d, l)
it_time = time() - it_time
print("[--] it time: {}".format(it_time))

it_time = time()
P.T
it_time = time() - it_time
print("[--] it time: {}".format(it_time))



