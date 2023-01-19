import torch
import time
import numpy as np

from eszd.directions import DirectionMatrix, GaussianDirections, SphericalDirections,  RadamacherDirections, RandomCoords, StructuredSphericalDirections

from utils import plot_comp_time, RotCoords, ReflectedDirections, ButterflyDirections, PermutedDirections

def compute_dmatrix_cost(dirs, reps=10):
    ctimes = []
    for _ in range(reps):
        g_time = time.time()
        dirs()
        g_time = time.time() - g_time
        ctimes.append(g_time)
    return np.mean(ctimes), np.std(ctimes)    
    

def get_time(dir_builders, dimensions, reps = 10):
    results = {}
    for (name, builder, params) in dir_builders:
        results[name] = []
        print("[{}] Started!".format(name))
        for l in dimensions:
            params['l'] = int(l)
            try:
            #    print(params)
                dirs = builder(**params)
                dirs()
                dirs()
                results[name].append(compute_dmatrix_cost(dirs, reps=reps))
            except RuntimeError as e:
                break
            print("\t[{}] Completed for l = {}".format(name, l))
        print("[{}] Completed!".format(name))
    return results

max_exp = 10

dimensions = [2, 1250, 2500, int(1e4/2)] #np.logspace(1, max_exp, max_exp)
seed = 121213
device = 'cpu'



common_params = {
    'd' : int(1e4),
    'l' : int(dimensions[0]),
    'seed' : seed,
    'device' : device,
    'dtype' : torch.float64
}

rot_params_id = {
    'd' : int(1e4),
    'l' : int(dimensions[0]),
    'seed' : seed,
    'device' : device,
    'use_identity' : True,
    'dtype' : torch.float64
}
rot_params_rand = {
    'd' : int(1e4),
    'l' : int(dimensions[0]),
    'seed' : seed,
    'device' : device,
    'use_identity' : False,
    'dtype' : torch.float64
}

builders = [
    ('gaussian', GaussianDirections, common_params),
    ('spherical', SphericalDirections, common_params),
    ('rademacher', RadamacherDirections, common_params),
    ('random coordinates', RandomCoords, common_params),
    ('RotCoords Id', RotCoords, rot_params_id),
    ('RotRandomCoords', RotCoords, rot_params_rand),
    ('Reflected', ReflectedDirections, common_params),
    ('Butterfly directions', ButterflyDirections, common_params),
    ('Permuted directions', PermutedDirections, common_params),
    ('QR directions', StructuredSphericalDirections, common_params),
]

reps= 5
results = get_time(builders, dimensions, reps = reps)
print(results)
for (label, result) in results.items():
    with open("./results_changing_l_{}.log".format(label), "w") as f:
        for (mu, std) in result:
            f.write("{},{}\n".format(mu, std))

out = "./changing_l"
plot_comp_time(results, dimensions, title="Time Comparison changing $\ell$ ($d = 10^4$)", xlab="$\ell$", xscale=None, out = out)
#mu_vec, std_vec = results['gaussian']
#print("[--] mu = {}\tstd = {}".format(mu_vec[0], std_vec[0]))


