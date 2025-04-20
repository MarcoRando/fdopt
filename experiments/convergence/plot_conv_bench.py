import sys
import numpy as np 

import matplotlib.pyplot as plt 


import pandas as pd 


#data_dir = "./convergence/ris/convergence/noise_0.0/ffd"
methods = ['ffd', 'cfd', 'opt_ffd', 'opt_cfd']
method = methods[0]
data_dir = f"./ris/convergence/noise_0.0/{method}"
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
      values += [float(splitted[1]) for _ in range(int(splitted[2]))]
      gammas.append(float(splitted[3]))
      if len(splitted) == 5:
        grad_norms.append(float(splitted[4]))
  return values, gammas, f_0, grad_norms


direction_types = ['gaussian', 'spherical', 'rademacher', 'coordinate', 'householder', 'butterfly', 'qr']
fun_names= ['Matyas', 'Booth', 'Schwefel_2_23_500', 'Schwefel_2_23_1000']
fun_labels= {
    'Matyas' : 'Matyas [$d = 2$]', 
    'Booth' : 'Booth [$d = 2$]', 
    'Schwefel_2_23_500' : 'Schwefel 2.23 [$d = 500$]',
    'Schwefel_2_23_1000' : 'Schwefel 2.23 [$d = 1000$]',
}

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()


for i in range(len(fun_names)):
  ax = axes[i]
  ax.set_title("{}".format(fun_labels[fun_names[i]]), fontsize=20)
  for dir_type in direction_type:
    mu_vals = []
    min_len = None
    for r in range(10):
      vals, _, f_0, _ = read_result(f"{data_dir}/{fun_names[i]}_{dir_type}_1e-07_{r}.csv")
      mu_vals.append(vals)
      if min_len is None or min_len > len(vals):
        min_len = len(vals)
    values = [np.array(elems[:min_len])   for elems in mu_vals]
    
    mu_vals = np.mean(values, axis=0)
    std_vals = np.std(values, axis=0)
    
    ax.plot(range(len(mu_vals)), mu_vals, '-', lw=3, label=dir_type, rasterized=True)
    ax.fill_between(range(len(mu_vals)), mu_vals - std_vals, mu_vals + std_vals, alpha=0.3, rasterized=True)

  ax.set_yscale('log')
  ax.set_xlabel("# Function Evaluations", fontsize=18)
#  ax.set_xscale('log')
  if i == 0:
    ax.legend(fontsize=12)
    ax.set_ylabel("$F(x_k)$", fontsize=18)
  elif i ==2:
    ax.set_ylabel("$F(x_k)$", fontsize=18)

fig.tight_layout()
fig.savefig(f"./linesearch_conv_bench_{method}.pdf", bbox_inches='tight')
plt.close(fig)


fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()


for i in range(len(fun_names)):
  ax = axes[i]
  ax.set_title("{}".format(fun_labels[fun_names[i]]), fontsize=20)
  for dir_type in direction_type:
    mu_vals = []
    min_len = None
    for r in range(10):
      vals, _, f_0, grads = read_result(f"{data_dir}/{fun_names[i]}_{dir_type}_1e-07_{r}.csv")
      mu_vals.append(grads)
      if min_len is None or min_len > len(grads):
        min_len = len(vals)
    values = [np.array(elems[:min_len])   for elems in mu_vals]
    
    mu_vals = np.mean(values, axis=0)
    std_vals = np.std(values, axis=0)
    
    ax.plot(range(len(mu_vals)), mu_vals, '-', lw=3, label=dir_type, rasterized=True)
    ax.fill_between(range(len(mu_vals)), mu_vals - std_vals, mu_vals + std_vals, alpha=0.3, rasterized=True)

  ax.set_yscale('log')
  ax.set_xlabel("$k$", fontsize=18)
#  ax.set_xscale('log')
  if i == 0:
    ax.legend(fontsize=12)
    ax.set_ylabel("$\\nabla F(x_k)$", fontsize=18)
  elif i ==2:
    ax.set_ylabel("$\\nabla F(x_k)$", fontsize=18)

fig.tight_layout()
fig.savefig(f"./linesearch_conv_bench_{method}_grad.pdf", bbox_inches='tight')
plt.close(fig)
