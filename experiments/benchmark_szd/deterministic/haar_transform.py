import torch
import numpy as np
from math import sqrt

sq2 = np.sqrt(2)
W2  = np.array([[1,1],[1,-1]])
H2  = W2/sq2
mat2mul = lambda M,v: [M[0][0]*v[0]+M[0][1]*v[1],M[1][0]*v[0]+M[1][1]*v[1]]
# mat2mul = lambda M,v: np.matmul(M,v)
hf1 = lambda data: np.array([mat2mul(H2,v2) for v2 in np.array(data).reshape(len(data)>>1,2)]).transpose().flatten()
def Haar(data):
    d1  = 1.0*np.array(data)
    k1,k2 = 1,len(data)
    while k2 > 1:
        d1 = d1.reshape(k1,k2)
        d1[0] = hf1(d1[0])
        k1 <<= 1
        k2 >>= 1
    return d1.flatten()

#print(Haar([1.0, 0.0, 0.0, 0.0]))
#print(Haar([0.0, 1.0, 0.0, 0.0]))

P = np.block([[Haar([1.0, 0.0, 0.0, 0.0])], [Haar([0.0, 1.0, 0.0, 0.0])]]).T
from torch_dct import dct
I = torch.eye(4, 2)
print(P, dct(I.T, norm='ortho').T)

P = torch.from_numpy(P)
print(dct(P.T, norm='ortho').T)
for _ in range(10):
    P = dct(P.T, norm='ortho').T
print(P)
def haar_transform(x : torch.Tensor):
    D = torch.empty(x.shape[0]//2, x.shape[1])
    X1 = torch.empty(x.shape[0]//2, x.shape[1])
    
    for i in range(x.shape[0]//2):
        X1[i] = 1/sqrt(2) * (x[i] + x[i+1])
        D[i] = 1/sqrt(2) * (x[i] - x[i+1])
    return X1, D
    
P = torch.eye(4, 2)

for _ in range(5):
    x, d = haar_transform(P.T)

    P = torch.hstack((x.T, d.T))

#print(P)
#print(P.T @ P)