import numpy as np
import torch
from numba.types import UniTuple
from numba import jit, f8,i8


def gmre(a, b):
    r = np.sqrt(a**2+b**2)
    return a/r, -b/r


def build_rotation_matrix(d, l, rnd_state):
    theta = 2*np.pi *rnd_state.rand()
    c, s = np.cos(theta), np.sin(theta)
    A = torch.eye(d)
    i, j = rnd_state.choice(d, 2, replace=False)
    i, j = (i, j) if i < j else (j, i)  
    A[i, i] = c    
    A[i, j] = -s    
    A[j, i] = s    
    A[j, j] = c 
    return A   

    
@jit(cache=True)
def theta_2_stiefel(A):
#    gen = torch.Generator()
#    gen.manual_seed(seed)
   # rnd_state = np.random.RandomState(seed)
   # A = rnd_state.randn(d, d)
    d, l = Z.shape
    n,p = d, l 
   # Z = np.eye(n)[:,:p]   # initialize orthonormal matrix as identity
    idx = -1              # initialize index of theta array
    nrm_c = np.sqrt(l/d)
    for i in range(p):
        for j in range(i+1, d):
            i, j = -i, -j
            # compute cos and sin of rotation angle
            r = np.sqrt(A[j, j]**2+ A[i, j]**2)
            cos, sin = A[j, j]/r, -A[i, j]/r

            #cos, sin = gmre(A[j, j], A[i, j]) #np.cos(theta[idx])
            # perform rotation on Z matrix
            for k in range(p):
                a = Z[i,k]
                b = Z[j,k]
                Z[i,k] = a*cos+b*sin
                Z[j,k] = -a*sin+b*cos
            
    return nrm_c * Z[:,:p]
    
def givens_qr(d, l, seed):
    gen = torch.Generator()
    gen.manual_seed(seed)
    A = torch.randn(d, d, generator=gen)

    Q = torch.eye(d)
    nrm_c = np.sqrt(l/d)
    for row in range(1, d):
        for col in range(0, np.min([row, l])):
            if A[row, col] != 0:
                (c, s) = gmre(A[col, col], A[row, col])
                G = torch.eye(d)
                G[col, col] = c
                G[row, row] = c
                G[row, col] = s
                G[col, row] = -s
                A = torch._sparse_mm(G,A)
                Q = torch._sparse_mm(Q, G.T)

    return nrm_c * Q.T[:, :l]