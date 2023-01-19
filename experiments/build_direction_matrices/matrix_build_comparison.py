from directions import CirculantDirections, SingleHouseholder, ButterflyDirections
import numpy as np
import torch
from torch_dct import dct

#P = ButterflyDirections(d=5, l=3)

x = torch.eye(2, 3)
mats = []
P_k =dct(x, norm='ortho')
for _ in range(1000):
    P_k = dct(P_k, norm='ortho')
    mats.append((P_k.T @ P_k).numpy())
print(np.mean(mats, 0))