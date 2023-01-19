import torch
from utils import GRot


d = 500
l = 2

P = GRot(d = d, l = l)

for _ in range(3):
    A = P()
    print(torch.allclose(A.T @ A, torch.eye(d)))

A = P()
print(A.T @ A)