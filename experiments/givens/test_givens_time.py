import torch
import numpy as np
from utils import givens_qr, theta_2_stiefel, build_rotation_matrix




d, l = 3, 3
I = np.sqrt(d/l) * torch.eye(d, l)
rnd_state = np.random.RandomState(1212)

G1 = build_rotation_matrix(d, l, rnd_state=rnd_state)
G2 = build_rotation_matrix(d, l, rnd_state=rnd_state)

A = I
#for _ in range(20):
G1 = build_rotation_matrix(d, l, rnd_state=rnd_state)
print(G1)

A = G1 @ torch.rand(d, l)
#A2 = G2 @ A
#print(np.sqrt(d/l) * A[:, 0].norm())
print (A.T @ A * l/d)