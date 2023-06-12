import sys
import torch
import numpy as np

sys.path.append("../")
sys.path.append("../targets/")
from utils import plot_comparison
from synthetic import LeastSquares

from fdopt import ZD
from fdopt.stepsize import ConstantStepsize, Stepsize
from fdopt.direction_generators import CoordinateDirections, GaussianDirections, SphericalDirections, RademacherDirections, QRDirections

seed = 12131415

dimensions = [2, 4, 5, 7, 8, 9, 10]# 12, 15, 20, 50, 100, 500]#[i for i in range(1, 6)] + [7, 10, 12, 15, 25, 50, 100, 500]
device = 'cpu'


def execute_experiment(d, P_class, gamma_0, f_min = 0.0, reps = 10, T = 1000):
    target = LeastSquares(m = d, d = d, device=device)
    P = P_class(d = d, l = d, seed=seed, device=device)
    x_0 = torch.ones((1, d), dtype=torch.float64, device=device)
    values = []
    traces = []
    for r in range(reps):
        L = target.get_L()
        gamma = ConstantStepsize(gamma_0=gamma_0 * 1/L)
        h = Stepsize(gamma_0=1.0, law=lambda t: (1/(t + 1)) ** 2)
        opt = ZD(gamma=gamma, h=h, P=P, g_estimator='forward')
        res = opt.optimize(f = target, x_0 = x_0, T = T, return_trace=True)
        values.append(res['fx'].item() - f_min)
        traces.append([x[1].item() - f_min for x in res['trace']])
    values = np.asarray(values).reshape(reps, -1)
    traces = np.asarray(traces).reshape(reps, T + 1)
    return values, traces
    

results = {}

labels = ["coordinate", "gaussian", "spherical", "qr"]
directions = [CoordinateDirections, GaussianDirections, SphericalDirections, QRDirections]

for (i, label) in enumerate(labels):
    results[label] = {}
    print(f"[++] {label}: Executing experiments...")
    for d in dimensions:
        print(f"\t[--] {label}: d = {d} completed!")
        gammas = [0.5, 0.5 * 1/d, 0.5, 0.5]
        values, traces = execute_experiment(d, directions[i], gammas[i], f_min = 0.0, reps=5, T = 1000)
        mu_fstar, std_fstar = values.mean(), values.std()
        mu_trace, std_trace = traces.mean(axis=0), traces.std(axis=0)
#        print(mu_trace.shape)
        results[label][d] = (mu_fstar, std_fstar, mu_trace, std_trace)

#print(results[labels[0]][2][2])
figsize = (10, 7)
for d in dimensions:
    mean_vals = np.asarray([results[label][d][2] for label in labels]).reshape(len(labels), -1)
    std_vals =  np.asarray([results[label][d][3] for label in labels]).reshape(len(labels), -1)
    plot_comparison(f"Function values d = {d}", mean_vals, std_vals, "$k$", "$f(x_k) - f(x^*)$", labels, f"comparison_d_{d}", figsize=figsize, plot_legend = False, yscale="log")
