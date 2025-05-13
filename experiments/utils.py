import time
import torch
import numpy as np
from fdopt.grad_estimators import ForwardFD, CentralFD, OptFFD, OptCFD
from fdopt.direction_generators import *#CoordinateDirections, GaussianDirections, SphericalDirections, Householder, ButterflyMatrix, QRDirections, SVDDirections

from synthetic_targets import *
from convex_benchmark import *
from nonconvex_benchmark import *

def get_grad_estimator(grad_estimator : str, norm_const : float = 1.0):
    if grad_estimator == 'ffd':
        return ForwardFD(norm_const=norm_const)
    elif grad_estimator == 'cfd':
        return CentralFD(norm_const=norm_const)
    elif grad_estimator == 'opt_ffd':
        return OptFFD(norm_const=norm_const)
    elif grad_estimator == 'opt_cfd':
        return OptCFD(norm_const=norm_const) 
    raise ValueError(f"Unknown gradient estimator {grad_estimator}!")       


def get_grad_cost(grad_estimator : str, l):
    if grad_estimator == 'ffd' or grad_estimator == 'opt_ffd':
        return l + 1
    elif grad_estimator == 'cfd' or grad_estimator=='opt_cfd':
        return 2 * l
    raise ValueError(f"Unknown gradient estimator {grad_estimator}!")       


def get_target_function(target_name):
    if target_name == 'least_squares':
        return LeastSquares
    elif target_name == 'logistic':
        return RegularizedLogistic
    elif target_name == 'trid':
        return TridFunction
    elif target_name == 'rosenbrock':
        return RosenbrockFunction
    elif target_name == 'quing':
        return QuingFunction
    elif target_name == 'stib':
        return StyblinskiTangFunction
    elif target_name == 'dixon_price':
        return ZakharovFunction
    raise Exception(f"Unknown target name: {target_name}")

direction_mapping = {
    'coordinate': CoordinateDirections,
    'gaussian': GaussianDirections,
    'spherical': SphericalDirections,
    'rademacher' : RademacherDirections,
    'householder': Householder,
    'butterfly': ButterflyMatrix,
    'qr': QRDirections,
}


def get_direction_generator(direction_generator: str):    
    if direction_generator in direction_mapping:
        return direction_mapping[direction_generator]
    raise ValueError(f"Invalid direction generator: {direction_generator}")

def relative_error(g, grad):
    return torch.linalg.norm(g.flatten() - grad.flatten(), ord=2) / torch.linalg.norm(grad.flatten(), ord=2)


def compute_error_time(fun, x, g, P, h, reps = 10):
    errors, times = [], []
    _ = g(fun, x, h = h, P = P()) # if x.shape[0] > 1 else
    for i in range(reps):
        grad = fun.grad(x).flatten()
        torch.cuda.synchronize() 
        it_time = time.time()
        P_inst = P()
        g_x = g(fun, x,  h = h, P = P_inst)
        torch.cuda.synchronize() 
        it_time = time.time() - it_time
        
        rel_err = relative_error(g_x, grad).item()
        errors.append(rel_err)
        times.append(it_time)
    return errors, times #np.mean(errors), np.std(errors), np.mean(times), np.std(times)


