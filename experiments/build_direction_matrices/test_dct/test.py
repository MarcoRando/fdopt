import torch
import time 
import numpy as np

#from utils import *
import torch_dct as dct
#data = np.eye(4, 2) #np.random.randn(4,4)

d = int(3)
P = torch.eye(1,d, device='cuda')

ldct = dct.LinearDCT(d, 'dct').to('cuda')
it_time = time.time()
P = dct.dct(P, norm='ortho')
#P = ldct(P)
it_time = time.time() - it_time
#
print(P)
for _ in range(10):
    P = dct.dct(dct.dct(P, norm='ortho'),norm='ortho')
    print(P)
print(P @ P.T)
print("[--] time: {}s".format(it_time))
#print(torch.allclose(P.T @ P, torch.eye(d, d, device='cuda'), atol=1e-3))
#P = PCA(data)
#print(torch.allclose(P.T @ P, torch.eye(d, d, device='cuda')))
#print(P.T @P)