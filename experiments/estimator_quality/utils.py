import torch


def relative_error(g, grad):
    return torch.linalg.norm(g- grad, ord=2) / torch.linalg.norm(grad)