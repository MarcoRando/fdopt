import torch

from math import log10

def compute_relative_error(grad, approx):
    approx = approx.reshape(-1)
    grad = grad.reshape(-1)
    return (torch.linalg.norm(approx - grad) / torch.linalg.norm(grad)).item()