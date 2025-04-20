import sys
import numpy as np 

import matplotlib.pyplot as plt 


import pandas as pd 


#data_dir = "./convergence/ris/convergence/noise_0.0/ffd"
methods = ['ffd', 'cfd', 'opt_ffd', 'opt_cfd']
method = methods[0]
data_dir = f"/data/mrando/fdopt_results/convergence/noise_0.0/{method}"
direction_type = ['spherical', 'gaussian', 'rademacher', 'coordinate', 'householder', 'butterfly',  'qr']

def read_result(fname):
  values, gammas = [], []
  grad_norms = []
  f_0 = None
  min_f = None
  with open(fname, 'r') as f:
    lines = f.readlines()
    for line in lines:
      splitted = line.split(',')
      if f_0 is None:
        f_0 = float(splitted[0])
        min_f = float(splitted[1])
      values.append(float(splitted[2]))# += [float(splitted[1]) for _ in range(int(splitted[2]))]
      gammas.append(float(splitted[4]))
      grad_norms.append(float(splitted[5])) #+= [float(splitted[4]) for _ in range(int(splitted[2]))]
  return values, gammas, f_0, min_f, grad_norms



def get_title_and_name(fun_name, dir_type, d, l, r):
  if fun_name == 'logistic':
    return "Logistic Regression", 'logreg', f"Logistic_{dir_type}_{l}_{d}_1e-07_{r}"
  elif fun_name == 'least_squares':
    return "Least Squares", 'ls', f"LeastSquare_{dir_type}_{l}_100000.0_1.0_{d}_1e-07_{r}"
  elif fun_name == 'trid':
    return "Trid", 'trid', f"Trid_{dir_type}_{l}_{d}_1e-07_{r}"
  elif fun_name == 'quing':
    return "Quing", 'quing', f'Quing_{dir_type}_{l}_{d}_1e-07_{r}'
  elif fun_name == 'rosenbrock':
    return "Rosenbrock", 'rosenbrock', f'Rosenbrock_{dir_type}_{l}_{d}_1e-07_{r}'
  elif fun_name == 'stib':
    return "Styblinski-Tang", 'stib', f'StyblinskiTang_{dir_type}_{l}_{d}_1e-07_{r}'
  raise ValueError(f"Unrecognized function {fun_name}")


fun_name = sys.argv[1]

num_dirs_label = ["1", "d/3", "d/2", "4/5 d", 'd']
dimensions = [100, 500, 1000]

fig, axes = plt.subplots(3, 5, figsize=(16, 10))

for (k, d) in enumerate(dimensions):
  ax = axes[k]
  num_directions = [1, d//3, d//2, round((4/5) * d),  d ]#[1, d//2,  round((4/5) * d), round((9/10) * d), d ]
  for i in range(len(num_directions)):
    ax[i].set_title("$\ell = $ {}".format(num_dirs_label[i]), fontsize=20)
    for dir_type in direction_type:
      mu_vals = []
      min_len = None
      for r in range(5):
        title, out_name, fname = get_title_and_name(fun_name, dir_type, d, num_directions[i], r)
        vals, _, f_0, min_f, _ = read_result(f"{data_dir}/{fname}.csv")      

        mu_vals.append(vals)
        if min_len is None or min_len > len(vals):
          min_len = len(vals)
      values = [(np.array(elems[:min_len]) - min_f) / max(elems[0] - min_f, 1e-20)    for elems in mu_vals]
      
      mu_vals = np.mean(values, axis=0)
      std_vals = np.std(values, axis=0)
      
      ax[i].plot(range(len(mu_vals)), mu_vals, '-', lw=3, label=dir_type, rasterized=True)
      ax[i].fill_between(range(len(mu_vals)), mu_vals - std_vals, mu_vals + std_vals, alpha=0.4, rasterized=True)

    ax[i].set_yscale('log')
#    ax[i].set_ylim([1e-2, 1.0])
    
    if k == 2:
        ax[i].set_xlabel("$k$", fontsize=16)
    if k == 0 and i ==0:
      ax[i].legend(fontsize=12)
    if i == 0:
      ax[i].set_ylabel("$\\frac{F(x_k) - \\min F}{F(x_0) - \\min F}$", fontsize=16)
fig.suptitle(title, fontsize=22)

fig.tight_layout()
fig.savefig(f"./linesearch_{out_name}.pdf", bbox_inches='tight')
plt.close(fig)


fig, axes = plt.subplots(3, 5, figsize=(16, 10))

fig.suptitle(title, fontsize=22)
for (k, d) in enumerate(dimensions):
  ax = axes[k]
  
  num_directions = [1, d//3, d//2, round((4/5) * d),  d ]#[1, d//2,  round((4/5) * d), round((9/10) * d), d ]
  for i in range(len(num_directions)):
    ax[i].set_title("$\ell = $ {}".format(num_dirs_label[i]), fontsize=20)
    for dir_type in direction_type:
      mu_vals = []
      min_len = None
      for r in range(5):
        title, out_name, fname = get_title_and_name(fun_name, dir_type, d, num_directions[i], r)
        vals, _, f_0, min_f, grad_norms = read_result(f"{data_dir}/{fname}.csv")      

        mu_vals.append(grad_norms)
        if min_len is None or min_len > len(grad_norms):
          min_len = len(grad_norms)
      values = [np.array(elems[:min_len])[:25] / elems[0]   for elems in mu_vals]
      
      mu_vals = np.mean(values, axis=0)
      std_vals = np.std(values, axis=0)
      
      ax[i].plot(range(len(mu_vals)), mu_vals, '-', lw=3, label=dir_type, rasterized=True)
      ax[i].fill_between(range(len(mu_vals)), mu_vals - std_vals, mu_vals + std_vals, alpha=0.4, rasterized=True)

    ax[i].set_yscale('log')
    if k == 2:
        ax[i].set_xlabel("$k$", fontsize=16)
    if k == 0 and i ==0:
      ax[i].legend(fontsize=12)
    if i == 0:
      ax[i].set_ylabel("$\\frac{\| \\nabla F(x_k) \|}{\| \\nabla F(x_0) \|}$", fontsize=16)

fig.tight_layout()
fig.savefig(f"./linesearch_{out_name}_grad.pdf", bbox_inches='tight')
plt.close(fig)
