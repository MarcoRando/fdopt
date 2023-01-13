import torch
import time
import numpy as np

from eszd.directions import DirectionMatrix, GaussianDirections, SphericalDirections, RadamacherDirections, RandomCoords, StructuredSphericalDirections

from utils import plot_comp_time

def compute_dmatrix_cost(dirs, reps=10):
    ctimes = []
    for _ in range(reps):
        g_time = time.time()
        dirs()
        g_time = time.time() - g_time
        ctimes.append(g_time)
    return np.mean(ctimes), np.std(ctimes)    
    

def get_time(dir_builders, params, dimensions, reps = 10):
    results = {}
    for (name, builder) in dir_builders:
        results[name] = []
        print("[{}] Started!".format(name))
        for d in dimensions:
            params['d'] = int(d)
            params['l'] = int(d)
            try:
                dirs = builder(**params)
                dirs()
                results[name].append(compute_dmatrix_cost(dirs, reps=reps))
            except RuntimeError:
                break
            print("\t[{}] Completed for d = {}".format(name, d))
        print("[{}] Completed!".format(name))
    return results

max_exp = 5

dimensions = np.logspace(1, max_exp, max_exp)
seed = 121213
device = 'cpu'


common_params = {
    'd' : int(dimensions[0]),
    'l' : int(dimensions[0]),
    'seed' : seed,
    'device' : device
}

builders = [
    ('gaussian', GaussianDirections),
    ('spherical', SphericalDirections),
    ('rademacher', RadamacherDirections),
    ('random coordinates', RandomCoords),
    ('QR directions', StructuredSphericalDirections),
]

reps= 5
results = get_time(builders, common_params, dimensions, reps = reps)
print(results)
for (label, result) in results.items():
    with open("./results_{}.log".format(label), "w") as f:
        for (mu, std) in result:
            f.write("{},{}\n".format(mu, std))

out = "./time_comp_standard"
plot_comp_time(results, dimensions, out = out)
#mu_vec, std_vec = results['gaussian']
#print("[--] mu = {}\tstd = {}".format(mu_vec[0], std_vec[0]))


