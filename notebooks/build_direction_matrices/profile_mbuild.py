import torch
from utils import RotCoords
import cProfile as cP
import pstats
import time
import numpy as np
from math import cos,sin, pi, sqrt, log2

from scipy.linalg import circulant, toeplitz, hankel, hadamard


import cProfile as cP
gen = torch.Generator()
gen.manual_seed(12)



def butterfly_prod(v, A):
    if v.shape[0] == 0:
    #    print("A base: ", A)
        return A
    n = A.shape[0]
    m = n // 2
    v1 = butterfly_prod(v[:-1], A[:m, :])
    v2 = butterfly_prod(v[:-1], A[m: , :])
    c, s = np.cos(v[-1]), np.sin(v[-1])
    #print("cv1: {}\tcv2: {}".format(c * v1 + s * v2, -s * v1 + c * v2))
    return torch.vstack([c * v1 + s * v2, -s * v1 + c * v2])

def sub_butterfly_prod(v, A, j, k):
    if k ==0:
        return butterfly_prod(v, A)
    if v.shape[0] == 0:
        return A
    n = A.shape[0]
    m = n // 2
    c, s = np.cos(v[-1]), np.sin(v[-1])
    if j > m:
        v1 = sub_butterfly_prod(v[:-1], A[:m, :], j - m, k -1)
        v2 = sub_butterfly_prod(v[:-1], A[m: , :], j - m, k - 1)
        return -s * v1 + c * v2
    #else:
    v1 = sub_butterfly_prod(v[:-1], A[:m, :], j, k -1)
    v2 = sub_butterfly_prod(v[:-1], A[m: , :], j, k - 1)
    return c * v1 + s * v2
    #print("cv1: {}\tcv2: {}".format(c * v1 + s * v2, -s * v1 + c * v2))
    return torch.vstack([c * v1 + s * v2, -s * v1 + c * v2])

@torch.jit.script
def bprod(v : torch.Tensor, A: torch.Tensor):
    width = 1    
    for i in range(v.shape[0]):# width < v.shape[0]:
        l = 0
        while l < A.shape[0]//width:# in range(v.shape[0]):
            a = l*width
            b = l*width + width
            c, s = cos(v[i]), sin(v[i])
            x1, x2 = A[a:b, :].clone(), A[b:b + width, :]
            A[a:b, :] = c *  x1 + s * x2
            A[b:b + width, :] = -s *  x1 + c * x2
            l+=2
        width*=2
            
    return A            


def butt_prod(A, l, n, thetas):
    if n ==0:
        return sqrt(A.shape[0]/l) * A[:, :l]
    theta = 2 * pi * thetas[-1]
    c, s = cos(theta), sin(theta)
    A_cut = A[:, :l]
    v1, v2 = torch.hstack([c * A_cut, - s * A_cut]), torch.hstack([s * A_cut, c * A_cut])
    block = torch.vstack([v1, v2])
    return butt_prod(block, l, n - 1, thetas[:-1])

def butt_prod_iter(n: int, l : int, thetas : torch.Tensor):
    A = torch.eye(1)
    lg_n = log2(n)
    k = int(lg_n) #+ 1 if lg_n % 1 > 0.0 else int(lg_n)
    num_additional = n - 2**k
  #  print("Num additional: ", num_additional) 
    for i in range(k):
        theta = 2 * pi * thetas[i]
        A_cut = A[:, :l]
        c, s = cos(theta) * A_cut, sin(theta) * A_cut
        #v1, v2 = torch.cat([c * A_cut, - s * A_cut], dim=1), torch.cat([s * A_cut, c * A_cut], dim=1)
#        v1, v2 = torch.hstack([c, - s]), torch.hstack([s, c])
        A = torch.hstack([c, -s, s, c]).reshape(2*c.shape[0], -1)
       # A = torch.vstack([v1, v2])
      #  A = torch.cat([v1, v2], dim=0 )
    A = A[:, :l]
  #  print("A shape: {}".format(A.shape))
  #  print(A)
    A = torch.vstack((A, torch.zeros(num_additional, A.shape[1])))#, dim=0)
    A = torch.hstack((A, torch.zeros(A.shape[0], num_additional)))#, dim=1)
   # print(A)
#    print(-i, list(range(-i, -1, -1)), A.shape)
    idx =range(-1, -num_additional -1, -1)
    A[idx, idx] = 2 * (torch.rand(num_additional) > 0.5).to(torch.float32) -1.0
  #  print(A)   
    #A[-i:, -i:].fill_diagonal_
#    for i in range(1, num_additional + 1):
#        A[-i, -i] = 2 * (torch.rand(1) > 0.5).to(torch.float32) -1.0
#    for i in range(num_additional):
#        A = torch.vstack((A, torch.zeros(1, A.shape[1])))
#        A = torch.hstack((A, torch.zeros(A.shape[0], 1)))
##        if i 
#        A[i, i] = 1.0
#    A[-1, -1]= 1.0
  #  print(A)
    inds = torch.randperm(A.shape[1])
    A = A[:, inds]
  #  print(A)
#    ginds = torch.ones(5)
#    P = torch.zeros((5, 5))
#    inds = ginds.multinomial(5, replacement=False)
#    P[inds, range(5)] = ((torch.rand(5) < 0.5)*2 - 1).to(torch.float32)
#    print("P: ",A.mm(P))
    return sqrt(A.shape[0]/l) * A[:, :l]#[]

    
gen = torch.Generator()
gen.manual_seed(12)

v = 2 * np.pi * np.random.rand(4)

#n = int(np.log2(1e4))
#v = 2 * torch.pi * torch.rand(n, generator=gen, dtype=torch.float32)
#A = torch.eye(2**n,2**n, dtype=torch.float32)
###import time
#_ = butterfly_prod(v, A)
#it_time = time.time()
#H = butterfly_prod(v, A)
#it_time = time.time() - it_time
#print("Recursive: ", it_time)
#print(H.shape)
##print("-----------------------")
#
##import cProfile as cP
#bprod(v, A.clone())
k = int(1e4)# int(np.log2(1e4))
l = k#int(np.log2(1e4))
thetas = 2 * pi * torch.rand(k)

#it_time = time.time()
#A = butt_prod(torch.ones((1, 1)), l, k, thetas) #butt_prod(torch.ones((1, 1)), 3, k, thetas)
#it_time = time.time() - it_time
#print("Recursive: ", it_time)

#it_time = time.time()
#A = butt_prod_iter(k, l, thetas) #butt_prod(torch.ones((1, 1)), 3, k, thetas)
#it_time = time.time() - it_time
#print("Iterative: ", it_time)
cP.run('butt_prod_iter(k, l, thetas)')
#print(A)
#print(A.T @ A * l/A.shape[0])
#print(A.T @ A * l/A.shape[0])
#print(np.allclose(A.T @ A, np.eye(A.shape[0])))
##
#print(torch.allclose(A, H))