import os
import time
import numpy as np
from fdopt.direction_generators import SphericalDirections,Householder,CoordinateDirections,GaussianDirections,QRDirections,ButterflyMatrix

out_dir = "./results/time_cost_bt"
os.makedirs(out_dir, exist_ok=True)
matrices = [
    ("butterfly", ButterflyMatrix),
    ("butterfly_dct", ButterflyMatrix)
]

def compute_cost(matrix_generator, reps=10):
    time_costs = []
    for _ in range(reps):
        it_time = time.time()
        _ = matrix_generator()
        it_time = time.time() - it_time
        time_costs.append(it_time)
    return np.mean(time_costs), np.std(time_costs)

dimensions = [1, 2, 3, 4, 5, 10, 12, 15, 20, 25, 50, 100, 150, 200, 500, 1000, 2000, 5000]#, 10000]

for d in dimensions:
    print(f"[++] Computing for d = {d}")
    with open("{}/time_costs_{}.log".format(out_dir,d), 'w') as f:
        for label, matrix_class in matrices:
            if label == 'butterfly_dct':
                mu, sigma = compute_cost(matrix_class(d = d, l = d, dct=True))
            else:
                mu, sigma = compute_cost(matrix_class(d = d, l = d))
            print("\t[{}] mu = {}\tsigma = {}".format(label, round(mu, 4), round(sigma, 4)))
            f.write("{},{},{}\n".format(label, mu, sigma))
            f.flush()



