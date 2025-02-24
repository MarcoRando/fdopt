
import os
import sys
import time
import tqdm
import torch
import pandas as pd 
import numpy as np
from fdopt.direction_generators import SphericalDirections,Householder,CoordinateDirections,GaussianDirections,QRDirections,ButterflyMatrix, SVDDirections


matrices = [
    ("spherical", SphericalDirections),
    ("householder", Householder),
    ("coordinate", CoordinateDirections),
    ("gaussian", GaussianDirections),
    ("butterfly", ButterflyMatrix),
    ("butterfly_dct", ButterflyMatrix),
    ("qr", QRDirections),
#    ("svd", SVDDirections),
]

def compute_cost(matrix_generator, reps=10):
    matrix_generator()
    time_costs = []
    for _ in range(reps):
#        if matrix_generator.device == 'cuda':
        torch.cuda.synchronize()
        it_time = time.time()
        matrix_generator()
#        if matrix_generator.device == 'cuda':
        torch.cuda.synchronize()
        it_time = time.time() - it_time
        time_costs.append(it_time)
    return np.mean(time_costs), np.std(time_costs)


devices = ['cuda', 'cpu']
dtype = torch.float64

dimensions = [1] + [(i + 1) * 100 for i in range(100)]#, 100, 500, 1000, 5000, 10000] #[int(x) for x in np.logspace()] #[1, 2, 10, 50, 100, 200, 500, 1000, 2000, 5000, 8000, 10000]
reps = sys.argv[1] # repetitions 500
out_dir = sys.argv[2] #"./results/time_cost"
os.makedirs(out_dir, exist_ok=True)


times_cpu, times_cuda = [], []

for device in devices:
    iterator = tqdm.tqdm(enumerate(dimensions))
    for (i, d) in iterator:
        for label, matrix_class in matrices:
            if label == 'butterfly_dct':
                mgen = matrix_class(d = d, l = d, dct=True, device=device, dtype=dtype)
            else:
                mgen = matrix_class(d = d, l = d, device=device, dtype=dtype)
            mu, sigma = compute_cost(mgen)
            if device == 'cuda':
                times_cuda.append((label, d, mu, sigma))
            else:
                times_cpu.append((label, d, mu, sigma))
            iterator.set_postfix({'d' : d, 'device' : device, 'directions' : label})
#            print("\t[{}] d = {}\tmu = {}\tsigma = {}".format(label, d, round(mu, 4), round(sigma, 4)))

for (time_data, dev) in zip([times_cpu, times_cuda], ['cpu', 'cuda']):
    dframe = pd.DataFrame(time_data, columns=['directions', 'd', 'mean_time', 'std_time'])
    dframe.to_csv(f"{out_dir}/time_{dev}.csv", index=False)

