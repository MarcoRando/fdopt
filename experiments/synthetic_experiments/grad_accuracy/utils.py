import torch

from math import log10

def compute_relative_error(grad, approx):
    return (torch.linalg.norm(approx - grad) / torch.linalg.norm(grad)).item()