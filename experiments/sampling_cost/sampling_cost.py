
import os
import sys
import time
import tqdm
import torch
import pandas as pd 
import numpy as np
from fdopt.direction_generators import * # SphericalDirections,Householder,CoordinateDirections,GaussianDirections,QRDirections,ButterflyMatrix, SVDDirections


matrices = [
    ("spherical", SphericalDirections),
    ("gaussian", GaussianDirections),
    ("rademacher", RademacherDirections),
    ("householder", Householder),
    ("permuted_householder", PermutedHouseholder),
    ("coordinate", CoordinateDirections),
    ("butterfly", ButterflyMatrix),
    ("subspace_qr", SubspaceQR),
    ("qr", QRDirections),
]

def compute_cost(matrix_generator, reps=10):
    matrix_generator()
    time_costs = []
    for _ in range(reps):
        torch.cuda.synchronize()
        it_time = time.time()
        matrix_generator()
        torch.cuda.synchronize()
        it_time = time.time() - it_time
        time_costs.append(it_time)
    return np.mean(time_costs), np.std(time_costs)

device = 'cuda'
seed = 12341567
dtype = torch.float64

dimensions = [500, 1000, 5000]#, 10000]

reps = sys.argv[1] 
out_dir = sys.argv[2] +"/sampling_cost"
os.makedirs(out_dir, exist_ok=True)


times_cuda = []

iterator = tqdm.tqdm(enumerate(dimensions))
for (i, d) in iterator:
    for l in [2, d//3, d//2, round((4/5) * d),  d]:
        for label, matrix_class in matrices:
            mgen = matrix_class(d = d, l = l, device=device, dtype=dtype, seed=seed)
            mu, sigma = compute_cost(mgen)
            times_cuda.append((label, d, l, mu, sigma))
            iterator.set_postfix({'d' : d, 'l' : l, 'directions' : label, 'mu' : mu, 'sigma' : sigma})

dframe = pd.DataFrame(times_cuda, columns=['directions', 'd', 'l', 'mean_time', 'std_time'])
dframe.to_csv(f"{out_dir}/cost_generating_matrices.csv", index=False)

