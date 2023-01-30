import torch
import numpy as np
from math import sqrt, log2

import time

#@torch.jit.script
def build_butterfly(block : torch.Tensor, thetas : torch.Tensor, l : int) -> torch.Tensor:
    if thetas.shape[0] == 0:
        return block[:, :l]
    theta = thetas[0]
    c, s = torch.cos(theta), torch.sin(theta)
    
    p1 = torch.vstack((c * block, s * block)) #np.block([[c * block, s * block], [-s * block, c * block]])
    p2 = torch.vstack((-s * block, c * block)) #np.block([[c * block, s * block], [-s * block, c * block]])

    block = torch.vstack((p1, p2))[:, :l]
    return build_butterfly(block, thetas=thetas[1:], l=l)


def build_block(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.block([[c , s], [-s, c]])

def get_block(thetas, n, m = 1, i = 0):
    R = build_block(thetas[i])
    cos_prod = 1.0
    sin_prod = 1.0
    for i in range(1, m):
        cos_prod *= np.cos(thetas[i])
    for j in range(m, n):
        sin_prod *= np.sin(thetas[j])
    return cos_prod * sin_prod * R


gen = torch.Generator()

gen.manual_seed(12121212)

    



#print("[--] angles: ", thetas)


#@torch.jit.script
def get_entry(i : int, j : int, thetas : torch.Tensor) -> float:
    if i == j:
        return thetas.cos().prod()
    vals = 1.0
    for k in range(0, len(thetas)):
        if (j >> k) % 2  == 0  and (i >> k) % 2  == 1:
            vals*=-1.0
        vals *= thetas[k].cos() if ((j >> k) + (i >> k) ) % 2 == 0 else thetas[k].sin()
    return vals

#@torch.jit.script
def build_parallel(n : int, l : int) -> torch.Tensor:
    thetas = 2 * torch.pi * torch.rand(n).to("cpu")
    d = int(2**n)
    P = torch.zeros(d, l, dtype=torch.float32, device='cpu')
    for i in range(P.shape[0]):
        for j in range(l):
            P[i, j] = get_entry(i, j, thetas)
            
    return sqrt(d/l) * P


def check_butt(B, thetas):
    k = len(thetas)
    print(B.shape)
    i = 4
    for j in range(B.shape[1]):
        print("[--] j: {}\tB[i, j] = {} => {}".format(j, B[i, j], get_entry(i, j, thetas)))


rnd_state = np.random.RandomState(1212)

thetas = 2 * torch.pi * torch.rand(int(log2(1e1))) #rnd_state.rand(int(log2(1e4)))

btime = time.time()
B = build_butterfly(torch.ones((1, 1)), thetas, 2)
btime = time.time() - btime

_ = build_parallel(1, 1)
it_time = time.time()
Bt = build_parallel(2, 2)
it_time = time.time() - it_time

print("[--] it time: {}".format(it_time))
print("[--] it time: {}".format(btime))
print(Bt.shape)

print(Bt.T @ Bt)
#check_butt(B, thetas)
#print("B: ", B)
#k = len(thetas)
#print(B)
##print(B_x[:4, 4:8])
##B_0_0 = 
#print(np.diag(B))
#print(np.cos(thetas).prod())
#print(- np.cos(thetas[2]) * np.sin(thetas[1]) * np.sin(thetas[0]))#np.sin(thetas).prod())
#print(B[1, 2])
#
#print(B)