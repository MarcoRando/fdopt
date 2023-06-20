import torch
import time
import numpy as np

from fdopt.grad_estimators import CentralGenStructGradApprox, CentralGenGradApprox, GenGradApprox, GenStructGradApprox, SinglePointEstimator, ForwardFD, CentralFD, LinInterpolation, CentralLinInterpolation
from fdopt.direction_generators import GaussianDirections, SphericalDirections, CoordinateDirections, QRDirections


from math import log10

def compute_relative_error(grad, approx):
    approx = approx.reshape(-1)
    grad = grad.reshape(-1)
    return (torch.linalg.norm(approx - grad) / torch.linalg.norm(grad)).item()


def get_methods(add_li = False):
    method_labels = {
        'FFD' : ForwardFD, 
        'CFD' : CentralFD, 
#        'SPE' : SinglePointEstimator,
        'GGA' : GenGradApprox,
        'CGGA' : CentralGenGradApprox
    }
    if add_li:
        method_labels['LI'] = LinInterpolation
        method_labels['CLI'] = CentralLinInterpolation
    direction_labels = {
        'G' : GaussianDirections, 
        'SP' : SphericalDirections, 
        'CO' : CoordinateDirections, 
        'QR' : QRDirections}
    methods = {}
    for (k_m, v_m) in method_labels.items():
        for (k_d, v_d) in direction_labels.items():
            methods[f"{k_m}_{k_d}"] = (v_m, v_d)
    return methods

def get_gsga_methods(methods):
    
    direction_labels = {
        'CO' : CoordinateDirections, 
        'QR' : QRDirections}
    for (k_d, dirs) in direction_labels.items():
        methods[f"GSGA_{k_d}"] = (GenStructGradApprox, dirs)
        methods[f"CGSGA_{k_d}"] = (CentralGenStructGradApprox, dirs)
    return methods

def get_g(target, method, d, l, device='cuda', dtype=torch.float64, gsga = False):
    (approximation, direction) = method
    if not gsga:
        P = direction(d = d, l = l, device = device, dtype = dtype, seed=12)
    else:
        P = direction(d = l, l = l, device = device, dtype = dtype, seed=12)
    return approximation(d = d, l = l, f = target, P = P, normalize=True)


def compute_grad_accuracy(targets, methods, d, l, h = 1e-10, reps = 10, device = 'cpu', dtype=torch.float64):
    # targets = [(target1, [x1]), (target2, [x1, x2, x3]), ...]
    results = {}
    for (target, xs) in targets:
        grad = target.grad
        results[repr(target)] = {}
        for (i, x) in enumerate(xs):
            results[repr(target)][repr(x.reshape(-1).tolist()) + f"_{i}"] = {}
            for (k_met, method) in methods.items():
                print(f"\t[{repr(target)}_{l}] testing {k_met} at x_{i}...")
                results[repr(target)][repr(x.reshape(-1).tolist()) + f"_{i}"][k_met] = {}
                if 'GSGA' not in k_met:
                    g = get_g(target, method, d, l, device = device, dtype = dtype, gsga=False) #if 'GSGA' not in k_met else get_gsga_g(target, method, d, device = device, dtype = dtype)
                else:
                    g = get_g(target, method, d, l, device = device, dtype = dtype, gsga=True)                    
                errs = []
                tm = []
                fx = target(x)
                gx = grad(x)
                for _ in range(reps):
                    g_time = time.time()
                    g_app = g(x, fx, h)
                    g_time = time.time() - g_time
                    err = compute_relative_error(gx, g_app)
                    errs.append(err)             
                    tm.append(g_time)       
                results[repr(target)][repr(x.reshape(-1).tolist()) + f"_{i}"][k_met] = (np.mean(errs), np.std(errs), np.mean(tm), np.std(tm))
    return results
    
            
def store_result(results, l, out_path, s = None):
    for (target_name, input_points) in results.items():
        out = "{}/{}_{}.log".format(out_path, target_name, l) if s is None else "{}/{}_{}_{}.log".format(out_path, target_name, l, s)
        with open(out, 'w') as f:
            for (k_x, methods) in input_points.items():
                f.write("[++] x = {}\n".format(k_x))
                for (met_name, err) in methods.items():
                    mu_err, std_err, mu_tm, std_tm = err
                    f.write("\t{},{},{},{},{}\n".format(met_name, mu_err, std_err, mu_tm, std_tm)) 

