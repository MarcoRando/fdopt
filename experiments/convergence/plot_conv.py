import sys
import numpy as np 

import matplotlib.pyplot as plt 


import pandas as pd 
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



fun_name = sys.argv[1]
d = int(sys.argv[2])

reps = int(sys.argv[3])

out_dir = sys.argv[4]

#T = int(sys.argv[5])


#data_dir = "./convergence/ris/convergence/noise_0.0/ffd"
method = 'ffd' 
data_dir = f"{out_dir}/convergence/noise_0.0/{method}"
direction_type = ['spherical', 'gaussian', 'rademacher', 'coordinate', 'householder', 'butterfly', 'permuted_householder', 'qr']
colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']
def read_result(fname):
  values, gammas = [], []
  grad_norms = []
  f_0, min_f = None, None
  with open(fname, 'r') as f:
    lines = f.readlines()
    for line in lines:
      splitted = line.split(',')
      if f_0 is None:
        f_0 = float(splitted[0])
        min_f = float(splitted[1])
      values += [float(splitted[2]) for _ in range(int(splitted[3]))]
#      values.append(float(splitted[2]))
      gammas.append(float(splitted[4]))
      # if len(splitted) == 5:
    #  grad_norms.append(float(splitted[5]))
  return values, gammas, f_0, min_f



if fun_name == 'logistic':
  title = "Logistic Regression"
  out_name = 'logreg'
elif fun_name == 'least_squares':
  title = "Least Squares"
  out_name = 'ls'
elif fun_name == 'sparse_sum':
  title = "Sparse Sum"
  out_name = 'sparse_sum'
elif fun_name == 'stib':
  title = "Styblinski-Tang"
  file_name = "StyblinskiTang"
  out_name = 'styblinski_tang'
elif fun_name == 'quing':
  title = "Qing"
  file_name = "Quing"
  out_name = 'quing'
elif fun_name == 'rosenbrock':
  title = "Rosenbrock"
  file_name = "Rosenbrock"
  out_name = 'rosenbrock'
elif fun_name == 'trid':
  title = "Trid"
  file_name = "Trid"
  out_name = 'trid'
elif fun_name == 'dixon_price':
  title = "Dixon Price"
  file_name = "DixonPrice"
  out_name = 'dixon_price'
elif fun_name == 'sum_of_powers':
  title = "Sum of Powers"
  file_name = "SumOfPowers"
  out_name = 'sum_of_powers'
elif fun_name == 'sum_of_squares':
  title = "Sum of Squares"
  file_name = "SumOfSquares"
  out_name = 'sum_of_squares'
elif fun_name == 'rotated_hyper_ellipsoid':
  title = "Rotated Hyper-Ellipsoid"
  file_name = "RotatedHyperEllipsoid"  
  out_name = 'rotated_hyper_ellipsoid'
elif fun_name == 'griewank':
  title = "Griewank"
  file_name = "Griewank"
  out_name = 'griewank'

num_directions = [2, d//3, d//2, round((4/5) * d),  d] #[1, d//2, round((2/3) * d), round((4/5) * d), d ]
num_dirs_label = ["2", "d/3", "d/2",  "4/5 d",  'd'] #["1", "d/2", "2/3 d", "4/5 d",  'd']


fig, axes = plt.subplots(1, 5, figsize=(20, 5))

#fig.suptitle("{} [$d = ${}]: {}".format(title, d, method.upper().replace("_","-")), fontsize=22)
fig.suptitle(title, fontsize=22)
axes = axes.flatten()


for i in range(len(num_directions)):
  ax = axes[i]
  ax.set_title("$\ell = $ {}".format(num_dirs_label[i]), fontsize=20)
  for (dir_id, dir_type) in enumerate(direction_type):
    mu_vals = []
    min_len = None
    for r in range(reps):
      if fun_name == 'logistic':
        vals, _, f_0,  min_f = read_result(f"{data_dir}/LogisticReg_1000_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")      
      elif fun_name == 'least_squares':
        vals, _, f_0,  min_f = read_result(f"{data_dir}/LeastSquares_10000.0_1.0_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")
      elif fun_name == 'sparse_sum':
        s = int(sys.argv[5])
        vals, _, f_0,  min_f = read_result(f"{data_dir}/SparseSum_{s}_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")
      else:
        vals, _, f_0,  min_f = read_result(f"{data_dir}/{file_name}_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")
      mu_vals.append(vals)
      if min_len is None or min_len > len(vals):
        min_len = len(vals)
    values = [(np.array(elems[:min_len]) - min_f) / (f_0 - min_f)   for elems in mu_vals]
    
    mu_vals = np.mean(values, axis=0)
    std_vals = np.std(values, axis=0) #- 
#    std_vals[std_vals < 1e-10] = 0.0
    print(dir_type, num_directions[i], mu_vals[-1], std_vals[-1])
    ax.plot(range(len(mu_vals)), mu_vals, '-', lw=3, label=dir_type.replace("_", " "), c=colors[dir_id], rasterized=True) 
    ax.fill_between(range(len(mu_vals)), mu_vals - std_vals, mu_vals + std_vals, alpha=0.3, color=colors[dir_id], rasterized=True)
  
  #ax.set_ylim([1e-20, 1.0])
  ax.set_yscale('log')
  ax.set_xlabel("# function evaluations", fontsize=16)
#  ax.set_xscale('log')
  if i == 0:
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylabel("$\\frac{F(x_k) - \\min F}{F(x_0) - \\min F}$", fontsize=18)
  # elif i ==3:
  #   ax.set_ylabel("$F(x_k)$", fontsize=18)

fig.tight_layout()
fig.savefig(f"./linesearch_{out_name}_{method}_{d}.pdf", bbox_inches='tight')
plt.close(fig)


# fig, axes = plt.subplots(1, 5, figsize=(18, 4))

# axes = axes.flatten()


# for i in range(len(num_directions)):
#   ax = axes[i]
#   ax.set_title("$\ell = $ {}".format(num_dirs_label[i]), fontsize=20)
#   for (dir_id, dir_type) in enumerate(direction_type):
#     mu_vals = []
#     min_len = None
#     for r in range(reps):
#       if fun_name == 'logistic':
#         vals, _, f_0, grad_norms, min_f = read_result(f"{data_dir}/LogisticReg_1000_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")      
#       elif fun_name == 'least_squares':
#         vals, _, f_0, grad_norms, min_f = read_result(f"{data_dir}/LeastSquares_10000.0_1.0_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")
#       elif fun_name == 'sparse_sum':
#         s = int(sys.argv[3])
#         vals, _, f_0, grad_norms, min_f = read_result(f"{data_dir}/SparseSum_{s}_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")
#       else:
#         vals, _, f_0, grad_norms, min_f = read_result(f"{data_dir}/{file_name}_{d}_{dir_type}_{num_directions[i]}_1e-07_{r}.csv")      
#       mu_vals.append(grad_norms)
#       if min_len is None or min_len > len(grad_norms):
#         min_len = len(grad_norms)
#     values = [np.array(elems[:min_len])    for elems in mu_vals]
#     mu_vals = np.mean(values, axis=0)
#     std_vals = np.std(values, axis=0)
    
#     ax.plot(range(len(mu_vals)), mu_vals, '-', lw=3, label=dir_type.replace("_", " "), c=colors[dir_id], rasterized=True)
#     ax.fill_between(range(len(mu_vals)), mu_vals - std_vals, mu_vals + std_vals, alpha=0.3, color=colors[dir_id], rasterized=True)

#   ax.set_yscale('log')
#   ax.set_xlabel("$k$", fontsize=18)
# #  ax.set_xscale('log')
#   if i == 0:
#     ax.legend(fontsize=12)
#     ax.set_ylabel("$\\frac{\\| \\nabla F(x_k) \\|}{\\| \\nabla F(x_0) \\|}$", fontsize=18)
#   # elif i ==3:
#   #   ax.set_ylabel("$\\| \\nabla F(x_k) \\|$", fontsize=18)

# fig.tight_layout()
# fig.savefig(f"./linesearch_{out_name}_{method}_{d}_grad.pdf", bbox_inches='tight')
# plt.close(fig)