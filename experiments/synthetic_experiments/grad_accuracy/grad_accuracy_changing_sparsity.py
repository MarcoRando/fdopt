import torch
import sys
import os
import numpy as np
from fdopt import ZD

from utils import compute_relative_error, get_g, get_methods, get_gsga_methods, compute_grad_accuracy, store_result

sys.path.append("../../targets/")
from synthetic import LeastSquares,SphereFunction, SparseSphere

device = 'cpu'
dtype = torch.float64
gen = torch.Generator(device=device)
gen.manual_seed(121212)

out_path = "../results/grad_accuracy/changing_sparsity/"
os.makedirs(out_path, exist_ok=True)            

L = 20
mu = 5
d = 50
sparsity = [1, 5, 10, 20, 30, 40, 45] #list(range(1, d))
#dimensions = [1, 2, 3, 5, 6, 7, 10, 20, 50, 100]#, 2, 3, 4, 5]#, 10, 20, 50, 100, 250, 450, 500]#, 10, 15, 20, 25, 30, 40, 50]#[i for i in range(2, 11)] #+ [15, 17, 20, 25, 30, 40, 50, 100, 200, 500]
reps = 200
h = 1e-10
with open(f"{out_path}/exp_info.log", 'w') as f:
    f.write(f"[--] L = {L}\n")
    f.write(f"[--] mu = {mu}\n")
    f.write(f"[--] d = {d}\n")
    f.write(f"[--] s = {repr(sparsity)}\n")
    f.write(f"[--] device = {device}\n")
    f.write(f"[--] dtype = {dtype}\n")
    f.write(f"[--] reps = {reps}\n")
    f.write(f"[--] h = {h}\n")
    
    
num_points = 7
for s in sparsity:
    t1 = SparseSphere(d = d, s = s, device = device, dtype = dtype) #LeastSquares(m = d, d=d, device=device, dtype=dtype, L = L, mu =mu)
    x1 = torch.ones((1, d), device=device, dtype=dtype)
    x2 = torch.tensor([i for i in range(d)], dtype=dtype, device=device).reshape(1, d) + 1e-10
    x3 = torch.tensor([i**2 + d for i in range(d)], device=device, dtype=dtype).reshape(1, d)
    xs = [x1, x2, x3] + [torch.rand(1, d, dtype=dtype, device=device, generator=gen) * 10 - 5 for _ in range(num_points)]
    print(f"[++] Executing experiment with d = {d} l = {d}")

    methods = get_methods(add_li=True)
    methods = get_gsga_methods(methods)
#    print(list(methods.keys()))
    objs = [(t1, xs)]

    results = compute_grad_accuracy(objs, methods,  d= d, l = d, h = h, reps=reps, device=device, dtype=dtype)

    store_result(results, d, out_path, s=s)

