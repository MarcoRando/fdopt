import torch

d = 5
l = 2

v = torch.randn(d)
v.div_(v.norm())

H = torch.eye(d, l) - 2 * v.outer(v[:l].T)

print(H)
print(H.T @ H)