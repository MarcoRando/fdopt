import torch
import numpy as np
from torch_dct import dct
from eszd.directions import Butterfly, HaarButterflyDirections, HBDirections

btt = HBDirections(d=1000, l=2, dtype=torch.float64)
A = btt()

print(A.T @ A)
#for i in range(10000):
#    btt()
#    print("i: {}/{}".format(i, 10000))
