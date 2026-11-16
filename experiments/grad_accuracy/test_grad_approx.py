import argparse
import csv
import datetime
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.append("../")
sys.path.append("../../")

# ---- adjust these three imports to the file names of your library ---------------------------------------------
from fdopt.directions.directions import FAMILIES  # direction operators
from fdopt.directions.directions import make as make_operator
from fdopt.optimizers.ffd import FFD  # FFD estimator
from synthetic_targets import (  # target functions
    GriewankFunction,
    LeastSquares,
    QuingFunction,
    RegularizedLogistic,
    RosenbrockFunction,
    TridFunction,
)

# ---------------------------------------------------------------------------------------------------------------

DTYPES = {"float64": torch.float64, "float32": torch.float32}
FAMILY_ORDER = ["coordinate", "qr", "householder", "permuted_householder", "butterfly",
                "gaussian", "spherical", "rademacher"]
STATS = ["mean", "std", "median", "p05", "p25", "p75", "p95", "min", "max"]
SEED = 121212
LOGISTIC_N = 1000          # number of samples of the RegularizedLogistic problem



target = LeastSquares(d=10, mu=1, L=1e6, dtype=torch.float64)

coord_dir = FAMILIES['coordinate'](d=10, l=10, device='cpu', dtype=torch.float64)

ffd = FFD(coord_dir, h=1e-7, norm_const=1.0)

x = torch.ones((1, 10), device='cpu', dtype=torch.float64)

grad = target.grad(x)

g_norm = torch.linalg.norm(grad)

g_app = ffd.estimate(target, x)


print("GERR: ", torch.linalg.norm(grad - g_app) / g_norm)
print("GERR flat: ", torch.linalg.norm(grad.flatten() - g_app.flatten()) / g_norm.flatten())

print(f"Grad: {grad}")
print(f"App: {g_app}")



