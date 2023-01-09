import torch
import time
import numpy as np
from sszd.direction_matrices import StructuredSphericalDirections
from utils import GSDirections,SRDirections

def get_time(generator, reps=10):
    _ = generator()
    gen_times = []
    for _ in range(reps):
        r_time = time.time()
        _ = generator()
        r_time = time.time() - r_time
        gen_times.append(r_time)
    return np.mean(gen_times), np.std(gen_times)
        
    
d, l = 3072, 3072
ss_time, ss_std = get_time(StructuredSphericalDirections(d, l, seed=12, dtype=torch.float64))
#sr_time, sr_std = get_time(SRDirections(d, l, seed=12, dtype=torch.float64))
#gs_time, gs_std = get_time(GSDirections(d, l, seed=12, dtype=torch.float64))

print("[--] Pytorch QR: {}s \pm {}s".format(ss_time, ss_std))
#print("[--] Schwarz-Rutishauser: {}s \pm {}s".format(sr_time, sr_std))
#print("[--] Gram-Schmidt: {}s \pm {}s".format(gs_time, gs_std))