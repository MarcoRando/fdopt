import torch
from utils import RotCoords, CirculantDirections
import cProfile as cP
import pstats
import time
import numpy as np
from math import cos,sin, pi, sqrt, log2


import cProfile as cP
gen = torch.Generator()
gen.manual_seed(12)
    

d = 3
l = 3

dirs = CirculantDirections(d=d, l=l)

print(dirs())