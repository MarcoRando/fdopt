from directions import CirculantDirections, SingleHouseholder, ButterflyDirections
from eszd.directions import GaussianDirections
import numpy as np
import torch
import time
import cProfile as cP

d = 10
l = 5
P = SingleHouseholder(d = 10, l=5, device='cuda')
#Pg = GaussianDirections(d = int(1e4), l=1000, device='cuda')
P()
P()
it_t = time.time()
P = P()
it_t = time.time() - it_t


print( P[:, 1].norm())

#cP.run("P()")
#cP.run("Pg()")
#print("[--] time: {}".format(it_t))

