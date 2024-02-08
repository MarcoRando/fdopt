import torch
import numpy as np
import time
from fdopt.direction_generators import OptimizedDirections, QRDirections, SphericalDirections, GaussianDirections, Householder

def compute_time(gen, reps = 10):
    times = []
    for _ in range(reps):
        it_time = time.time()
        _ = gen()
        it_time = time.time() - it_time
        times.append(it_time)
    times = np.array(times)
    return np.mean(times), np.std(times)



dtype = torch.float32
device = 'cpu'
seed = 121314
reps = 1

for d in [int(5e3)]:#[2**i for i in range(1, 12)]:
    Opt = OptimizedDirections(eps=1e-4, gamma=1.0, d=d, l =d, dtype=dtype, device=device, seed=seed)
    QR = QRDirections(d = d, l = d, dtype=dtype, device=device, seed=seed)
    Sph = SphericalDirections(d = d, l = d, dtype=dtype, device=device, seed=seed)
    H = Householder(d = d, l = d, num_reflectors=3, dtype=dtype, device=device, seed=seed)
    generators = [
        (H, "Householder"),
        (QR, "QR"),
        #(Opt, "Optimized"),
        (Sph, "Spherical"),
    ]
    print("[--] Computing for d = {}".format(d))
    for (gen, label) in generators:
        mu, sigma = compute_time(gen, reps = reps)
        
        print("\t[{}] elapsed time = {} +/- {}".format(label, mu, sigma))

