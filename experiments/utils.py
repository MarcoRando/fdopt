import torch
import time
import numpy as np
from fdopt.grad_estimators import ForwardFD, CentralFD, OptFFD, OptCFD
from fdopt.direction_generators import CoordinateDirections, GaussianDirections, SphericalDirections, Householder, ButterflyMatrix, QRDirections, SVDDirections

from synthetic_targets import *

def relative_error(g, grad):
    return torch.linalg.norm(g.flatten() - grad.flatten(), ord=2) / torch.linalg.norm(grad.flatten(), ord=2)


def get_grad_estimator(grad_estimator : str, norm_const : float = 1.0):
    if grad_estimator == 'ffd':
        return ForwardFD(norm_const=norm_const)
    elif grad_estimator == 'cfd':
        return CentralFD(norm_const=norm_const)
    elif grad_estimator == 'opt_ffd':
        return OptFFD(norm_const=norm_const)
    elif grad_estimator == 'opt_cfd':
        return OptCFD(norm_const=norm_const) 
    raise ValueError("Unknown gradient estimator!")       

def get_target_function(target_name, **params):
    if target_name == "ls1":
        return LS1(**params)
    elif target_name == 'ls2':
        return LS2(**params)
    elif target_name == 'ls3':
        return LS3(**params)
    return PycutestTarget(problem_name=target_name, **params)
    raise Exception(f"Unknown target name: {target_name}")

direction_mapping = {
    'coordinate': CoordinateDirections,
    'gaussian': GaussianDirections,
    'spherical': SphericalDirections,
    'householder': Householder,
    'butterfly': ButterflyMatrix,
    'butterfly_dct': ButterflyMatrix,
    'qr': QRDirections,
    'svd': SVDDirections
}


def get_direction_generator(direction_generator: str):    
    if direction_generator in direction_mapping:
        return direction_mapping[direction_generator]
    raise ValueError(f"Invalid direction generator: {direction_generator}")

def compute_error_time(fun, x, fx, g, P, h, reps = 10):
    errors, times = [], []
    grad = fun.grad(x).flatten()
    _ = g(fun, x, f_x = fx, h = h, P = P())
    for _ in range(reps):        
        it_time = time.time()
        g_x = g(fun, x, f_x = fx, h = h, P = P())
        it_time = time.time() - it_time
        print("G", g_x)
        print("GRAD:", grad)
        exit()
        rel_err = relative_error(g_x, grad).item()
        errors.append(rel_err)
        times.append(it_time)
    return np.mean(errors), np.std(errors), np.mean(times), np.std(times)

