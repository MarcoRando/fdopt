import torch
import sys
import os
import numpy as np
from fdopt import ZD

from utils import compute_relative_error, get_g, get_methods, get_gsga_methods, compute_grad_accuracy, store_result

sys.path.append("../../targets/")
from synthetic import LeastSquares,SphereFunction

device = 'cpu'
dtype = torch.float64


out_path = "../results/grad_accuracy/changing_d/"
os.makedirs(out_path, exist_ok=True)            

L = 20
mu = 5
dimensions =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100, 500, 600, 1000]#, 2, 3, 4, 5]#, 10, 20, 50, 100, 250, 450, 500]#, 10, 15, 20, 25, 30, 40, 50]#[i for i in range(2, 11)] #+ [15, 17, 20, 25, 30, 40, 50, 100, 200, 500]
reps = 200
gen = torch.Generator(device=device)
gen.manual_seed(121212)

h = 1e-10
with open(f"{out_path}/exp_info.log", 'w') as f:
    f.write(f"[--] L = {L}\n")
    f.write(f"[--] mu = {mu}\n")
    f.write(f"[--] dimensions = {repr(dimensions)}\n")
    f.write(f"[--] device = {device}\n")
    f.write(f"[--] dtype = {dtype}\n")
    f.write(f"[--] reps = {reps}\n")
    f.write(f"[--] h = {h}\n")
    
x_max, x_min = 10.0, 2.0
for d in dimensions:
    t1 = SphereFunction(d = d, device=device, dtype=dtype) #LeastSquares(m = d, d=d, device=device, dtype=dtype, L = L, mu =mu)
#    x = torch.ones((1, d), device=device, dtype=dtype)
#    x = torch.tensor([i + 1 for i in range(d)], dtype=dtype, device=device).reshape(1, d) + 1e-10
    x = (x_max - x_min) * torch.rand((1, d), device=device, dtype=dtype, generator=gen) + x_min
#    x = torch.full((1, d), 12.0, dtype=dtype, device=device)
    print(f"[++] Executing experiment with d = {d} l = {d}")

    methods = get_methods(add_li=True)
    methods = get_gsga_methods(methods)
#    print(list(methods.keys()))
    objs = [(t1, [x])]

    results = compute_grad_accuracy(objs, methods,  d= d, l = d, h = h, reps=reps, device=device, dtype=dtype)

    store_result(results, d, out_path)

