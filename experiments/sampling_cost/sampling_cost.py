
import os
import time
import tqdm
import torch
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
    ("svd", SVDDirections),
]

def compute_cost(matrix_generator, reps=10):
    matrix_generator()
    time_costs = []
    for _ in range(reps):
        it_time = time.time()
        matrix_generator()
        it_time = time.time() - it_time
        time_costs.append(it_time)
    return np.mean(time_costs), np.std(time_costs)


devices = ['cuda', 'cpu']
dtypes = [torch.float32, torch.float64]

dimensions = [1, 10, 50, 100, 500, 1000, 5000, 10000] #[int(x) for x in np.logspace()] #[1, 2, 10, 50, 100, 200, 500, 1000, 2000, 5000, 8000, 10000]
reps = 500
out_dir ="./results/time_cost"
os.makedirs(out_dir, exist_ok=True)

iterator = tqdm.tqdm(enumerate(dimensions))


for (i, d) in iterator:
#    print(f"[++] Computing for d = {d}")
    iterator.set_postfix({'d' : d})
    for device in devices:
        for dtype in dtypes:
            with open("{}/time_costs_{}_{}_{}.log".format(out_dir, d, device, dtype), 'w') as f:
                for label, matrix_class in matrices:
                    if label == 'butterfly_dct':
                        mgen = matrix_class(d = d, l = d, dct=True, device=device, dtype=dtype)
                    else:
                        mgen = matrix_class(d = d, l = d, device=device, dtype=dtype)
                    mu, sigma = compute_cost(mgen)
                    print("\t[{}] mu = {}\tsigma = {}".format(label, round(mu, 4), round(sigma, 4)))
                    f.write("{},{},{}\n".format(label, mu, sigma))
                    f.flush()



