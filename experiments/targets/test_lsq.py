import torch

from synthetic import LeastSquares

m = d= 3

fun = LeastSquares(m, d)
x = torch.ones(1, d, dtype=torch.float64)


print(fun(x))