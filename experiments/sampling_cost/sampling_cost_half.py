import os
import time
import numpy as np
from fdopt.direction_generators import SphericalDirections,Householder,CoordinateDirections,GaussianDirections,QRDirections,ButterflyMatrix


matrices = [
    ("spherical", SphericalDirections),
    ("householder", Householder),
    ("coordinate", CoordinateDirections),
    ("gaussian", GaussianDirections),
    ("qr", QRDirections),
    ("butterfly", ButterflyMatrix),
#    ("butterfly_dct", ButterflyMatrix)
]
out_dir = "./results/time_cost_dhalf"
os.makedirs(out_dir, exist_ok=True)

def compute_cost(matrix_generator, reps=100):
    time_costs = []
    for _ in range(reps):
        it_time = time.time()
        _ = matrix_generator()
        it_time = time.time() - it_time
        time_costs.append(it_time)
    return np.mean(time_costs), np.std(time_costs)

dimensions = [20, 50, 100, 500, 1000, 2000]#, 10000]

num_directions = [lambda d : 1, lambda d : d // 4, lambda d : d // 2, lambda d : d - 1]

for d in dimensions:
    for num_dir in num_directions:
        l = num_dir(d)
        print(f"[++] Computing for d = {d}\tl = {l}")
        with open("{}/time_costs_{}_{}.log".format(out_dir, d, l), 'w') as f:
            for label, matrix_class in matrices:
                if label == 'butterfly_dct':
                    mu, sigma = compute_cost(matrix_class(d = d, l = l, dct=True))
                else:
                    mu, sigma = compute_cost(matrix_class(d = d, l = l))
                print("\t[{}] mu = {}\tsigma = {}".format(label, round(mu, 4), round(sigma, 4)))
                f.write("{},{},{}\n".format(label, mu, sigma))
                f.flush()



