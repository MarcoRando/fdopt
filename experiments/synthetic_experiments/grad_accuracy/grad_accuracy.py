import torch
import sys
import os
import numpy as np
from fdopt import ZD
from fdopt.grad_estimators import GenGradApprox, GenStructGradApprox, SinglePointEstimator, ForwardFD, CentralFD, LinInterpolation, CentralLinInterpolation
from fdopt.direction_generators import GaussianDirections, SphericalDirections, CoordinateDirections, QRDirections

from fdopt.stepsize import ConstantStepsize

from utils import compute_relative_error

sys.path.append("../../targets/")
from synthetic import LeastSquares

device = 'cuda'
dtype = torch.float64


def get_methods():
    directions = [GaussianDirections, SphericalDirections, CoordinateDirections, QRDirections]
    estimators = [ForwardFD, CentralFD, LinInterpolation, CentralLinInterpolation, GenGradApprox] 
    method_labels = {
        'FFD' : ForwardFD, 
        'CFD' : CentralFD, 
        'SPE' : SinglePointEstimator,
        'LI' : LinInterpolation, 
        'CLI' : CentralLinInterpolation,
        'GGA' : GenGradApprox
    }
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
    return methods

def get_g(target, method, d, device='cuda', dtype=torch.float64):
    (approximation, direction) = method
    P = direction(d = d, l = d, device = device, dtype = dtype)
    return approximation(d = d, l = d, f = target, P = P, normalize=True)

def compute_grad_accuracy(targets, methods, h = 1e-10, reps = 10):
    # targets = [(target1, [x1]), (target2, [x1, x2, x3]), ...]
    results = {}
    for (target, xs) in targets:
        d = target.d
        grad = target.grad
        results[repr(target)] = {}
        for x in xs:
            results[repr(target)][repr(x.reshape(-1).tolist())] = {}
            for (k_met, method) in methods.items():
                print(f"\t[{repr(target)}] testing {k_met} at {repr(x.reshape(-1).tolist())}...")
                results[repr(target)][repr(x.reshape(-1).tolist())][k_met] = {}
                g = get_g(target, method, d, device = device, dtype = dtype) #if 'GSGA' not in k_met else get_gsga_g(target, method, d, device = device, dtype = dtype)
                errs = []
                for _ in range(reps):
                    errs.append(compute_relative_error(grad(x), g(x, target(x), h)))                    
                results[repr(target)][repr(x.reshape(-1).tolist())][k_met] = (np.mean(errs), np.std(errs))
    return results
    
            
def store_result(results):
    for (target_name, input_points) in results.items():
        with open("{}/{}.log".format(out_path, target_name), 'w') as f:
            for (k_x, methods) in input_points.items():
                f.write("[++] x = {}\n".format(k_x))
                for (met_name, err) in methods.items():
                    mu, std = err
                    f.write("\t{},{},{}\n".format(met_name, mu, std)) 


out_path = "../results/grad_accuracy/changing_d/"
os.makedirs(out_path, exist_ok=True)            
device = 'cpu'
dtype = torch.float64

L = 20
mu = 1
dimensions = [i for i in range(2, 11)] #+ [15, 17, 20, 25, 30, 40, 50, 100, 200, 500]
reps = 200
h = 1e-10
with open(f"{out_path}/exp_info.log", 'w') as f:
    f.write(f"[--] L = {L}\n")
    f.write(f"[--] mu = {mu}\n")
    f.write(f"[--] dimensions = {repr(dimensions)}\n")
    f.write(f"[--] device = {device}\n")
    f.write(f"[--] dtype = {dtype}\n")
    f.write(f"[--] reps = {reps}\n")
    f.write(f"[--] h = {h}\n")

for d in dimensions:
    print(f"[++] Executing experiment with d = {d}")
    target = LeastSquares(m = d, d=d, device=device, dtype=dtype, L = L, mu =mu)
    x1 = torch.ones((1, d), device=device, dtype=dtype)
    x2 = torch.zeros((1, d), device=device, dtype=dtype)
    x3 = torch.tensor([i for i in range(d)], dtype=dtype, device=device) + 1e-10
    x4 = torch.tensor([i**2 + d for i in range(d)], device=device, dtype=dtype)
    x5 = torch.full((1, d), 12.0, device=device, dtype=dtype)

    methods = get_methods()
    methods = get_gsga_methods(methods)
    targets = [(target, [x1, x2, x3, x4, x5])]

    results = compute_grad_accuracy(targets, methods, h = h, reps=reps)

    store_result(results)


#print(results)
#x = torch.ones((1, d), device=device, dtype=dtype)
#
#P = QRDirections(d = d, l = d, device=device, dtype=dtype)
#g_est = ForwardFD(target, P = P, normalize=True)
#
#h = 1e-10
#fx = target(x)
#
#
#print(compute_relative_error(grad(x), g_est(x, fx, h)))


