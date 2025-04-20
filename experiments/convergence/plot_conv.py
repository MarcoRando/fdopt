import sys
import numpy as np 

import matplotlib.pyplot as plt 


import pandas as pd 


#data_dir = "./convergence/ris/convergence/noise_0.0/ffd"
methods = ['ffd', 'cfd', 'opt_ffd', 'opt_cfd']
method = methods[0]
data_dir = f"./convergence/noise_0.0/{method}"
direction_type = ['spherical', 'gaussian', 'rademacher', 'coordinate', 'householder', 'butterfly',  'qr']

def read_result(fname):
  values, gammas = [], []
  grad_norms = []
  f_0 = None
  with open(fname, 'r') as f:
    lines = f.readlines()
    for line in lines:
      splitted = line.split(',')
      if f_0 is None:
        f_0 = float(splitted[0])
      values += [float(splitted[2]) for _ in range(int(splitted[3]))]
      gammas.append(float(splitted[4]))
      # if len(splitted) == 5:
      grad_norms.append(float(splitted[5]))
  return values[:4000], gammas, f_0, grad_norms[:4000]


fun_name = sys.argv[1]
d = int(sys.argv[2])

if fun_name == 'logistic':
    title = "Logistic Regression"
    out_name = 'logreg'
elif fun_name == 'least_squares':
    title = "Least Squares"
    out_name = 'ls'



num_directions = [1, d//3, d//2, round((4/5) * d),  d] #[1, d//2, round((2/3) * d), round((4/5) * d), d ]
num_dirs_label = ["1", "d/3", "d/2",  "4/5 d",  'd'] #["1", "d/2", "2/3 d", "4/5 d",  'd']


fig, axes = plt.subplots(1, 5, figsize=(18, 4))

fig.suptitle("{} [$d = ${}]: {}".format(title, d, method.upper().replace("_","-")), fontsize=22)
axes = axes.flatten()


for i in range(len(num_directions)):
  ax = axes[i]
  ax.set_title("$\ell = $ {}".format(num_dirs_label[i]), fontsize=20)
  for dir_type in direction_type:
    mu_vals = []
    min_len = None
    for r in range(5):
      if fun_name == 'logistic':
        vals, _, f_0, _ = read_result(f"{data_dir}/LogisticReg_1000_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")      
      else:
        vals, _, f_0, _ = read_result(f"{data_dir}/LeastSquares_10000.0_1.0_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")
      mu_vals.append(vals)
      if min_len is None or min_len > len(vals):
        min_len = len(vals)
    values = [np.array(elems[:min_len])[:10000]   for elems in mu_vals]
    
    mu_vals = np.mean(values, axis=0)
    std_vals = np.std(values, axis=0)
    
    ax.plot(range(len(mu_vals)), mu_vals, '-', lw=3, label=dir_type, rasterized=True)
    ax.fill_between(range(len(mu_vals)), mu_vals - std_vals, mu_vals + std_vals, alpha=0.4, rasterized=True)

  ax.set_yscale('log')
  ax.set_xlabel("# Function Evaluations", fontsize=18)
#  ax.set_xscale('log')
  if i == 0:
    ax.legend(fontsize=12)
    ax.set_ylabel("$F(x_k)$", fontsize=18)
  # elif i ==3:
  #   ax.set_ylabel("$F(x_k)$", fontsize=18)

fig.tight_layout()
fig.savefig(f"./linesearch_{out_name}_{method}_{d}.pdf", bbox_inches='tight')
plt.close(fig)


fig, axes = plt.subplots(1, 5, figsize=(18, 4))

fig.suptitle("{} [$d = ${}]: {}".format(title, d, method.upper().replace("_","-")), fontsize=22)
axes = axes.flatten()


for i in range(len(num_directions)):
  ax = axes[i]
  ax.set_title("$\ell = $ {}".format(num_dirs_label[i]), fontsize=20)
  for dir_type in direction_type:
    mu_vals = []
    min_len = None
    for r in range(5):
      if fun_name == 'logistic':
        vals, _, f_0, grad_norms = read_result(f"{data_dir}/LogisticReg_1000_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")      
      else:
        vals, _, f_0, grad_norms = read_result(f"{data_dir}/LeastSquares_10000.0_1.0_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")
      mu_vals.append(grad_norms)
      if min_len is None or min_len > len(grad_norms):
        min_len = len(grad_norms)
    values = [np.array(elems[:min_len])   for elems in mu_vals]
    mu_vals = np.mean(values, axis=0)
    std_vals = np.std(values, axis=0)
    
    ax.plot(range(len(mu_vals)), mu_vals, '-', lw=3, label=dir_type, rasterized=True)
    ax.fill_between(range(len(mu_vals)), mu_vals - std_vals, mu_vals + std_vals, alpha=0.4, rasterized=True)

  ax.set_yscale('log')
  ax.set_xlabel("$k$", fontsize=18)
#  ax.set_xscale('log')
  if i == 0:
    ax.legend(fontsize=12)
    ax.set_ylabel("$\\| \\nabla F(x_k) \\|$", fontsize=18)
  # elif i ==3:
  #   ax.set_ylabel("$\\| \\nabla F(x_k) \\|$", fontsize=18)

fig.tight_layout()
fig.savefig(f"./linesearch_{out_name}_{method}_{d}_grad.pdf", bbox_inches='tight')
plt.close(fig)