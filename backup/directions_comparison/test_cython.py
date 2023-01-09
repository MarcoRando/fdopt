from gram_schmidt import gram_schmidt

import torch

import array

A = torch.ones((3, 2))


ris = gram_schmidt(A.numpy())

print(ris)