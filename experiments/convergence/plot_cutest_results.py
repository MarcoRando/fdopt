import sys
import numpy as np 
import os

import matplotlib.pyplot as plt 


import pandas as pd 
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

base_dir = sys.argv[1]
reps = int(sys.argv[2])
#method = 'ffd'
data_dir = f"{base_dir}/ffd/cutest"
direction_type = ['spherical', 'gaussian', 'rademacher', 'coordinate', 'householder', 'butterfly', 'permuted_householder', 'qr']
colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']
num_dirs_label = ["2", "d/3", "d/2",  "4/5 d",  'd'] #["1", "d/2", "2/3 d", "4/5 d",  'd']



# def read_result(fname):
#   values, gammas = [], []
#   grad_norms = []
#   f_0, min_f = None, None
#   with open(fname, 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#       splitted = line.split(',')
#       if f_0 is None:
#         f_0 = float(splitted[0])
#         min_f = float(splitted[1])
# #      values += [float(splitted[2]) for _ in range(int(splitted[3]))]
#       values.append(float(splitted[2]))
#       gammas.append(float(splitted[4]))
#       grad_norms.append(float(splitted[5]))
#   return values, gammas, f_0, min_f

# def get_pycutest_target(problem_name):
    
#     splitted = problem_name.split(',')
    
#     name = splitted[0]
#     d = int(splitted[1])
#     n = int(splitted[2])
    
#     x = np.array([float(elem) for elem in splitted[3:]])
    
#     problem = pc.import_problem(name.split("_")[0], sifParams={'N' : d})
#     return PycutestTarget(problem, dtype=dtype, device=device, seed = seed, x0=x), d, name

problems = []

with open("../cutest_problem_names.log", 'r') as f:
    for line in f.readlines():
      splitted = line.split(',')
      problems.append((splitted[0], int(splitted[1]), int(splitted[2])))
#      print(line)
#      exit()
directories = os.listdir(f"{base_dir}/ffd/cutest")

print("[--] Num files: ", len(directories))

#problems = [("ARGLINA", 200, 200)]

min_values = {}
init_values = {}
values = {}
for (function, d_idx, d) in problems:
  num_directions = [2, d//3, d//2, round((4/5) * d),  d] #[1, d//2, round((2/3) * d), round((4/5) * d), d ]
  if f"{function}_{d_idx}_{d}" not in min_values:
    min_values[f"{function}_{d_idx}_{d}"] = None
  if f"{function}_{d_idx}_{d}" not in values:
    values[f"{function}_{d_idx}_{d}"] = {dir_type : [] for dir_type in direction_type}

  if f"{function}_{d_idx}_{d}" not in init_values:
    init_values[f"{function}_{d_idx}_{d}"] = None
  else:
    print("ALREADY EXISTS")
    exit()
  for dir_type in direction_type:
    if f"{function}_{d_idx}_{d}" not in min_values:
      break
    for (i, l) in enumerate(num_directions):
      dir_values = []
      min_v = None
      for r in range(reps):
        if os.path.exists(f"{base_dir}/ffd/cutest/{function}_{d_idx}_{d}_{dir_type}_{l}_1e-07_{r}.csv"):
          data = pd.read_csv(f"{base_dir}/ffd/cutest/{function}_{d_idx}_{d}_{dir_type}_{l}_1e-07_{r}.csv", header=None)
          val_iters = data[2].values
          feval_val = []
          for (val_idx, val) in enumerate(val_iters[:-1]):
            feval_val += [val for _ in range(data[3].values[val_idx + 1])]
          feval_val += [val_iters[-1]]

          dir_values.append(feval_val)

          min_val = data[1].values[0]
          f0 = data[0].values[0]
          if init_values[f"{function}_{d_idx}_{d}"] is None:
            print(f"[--] Loading results for {function}_{d_idx}_{d}")
            init_values[f"{function}_{d_idx}_{d}"] = f0
          if min_v is None or min_val < min_v:
            min_v = min_val
#        else:
#          print(f"[--] Skipping {function}_{d_idx}_{d} since it is not in the results")
      if min_v is not None:
        min_len = min([len(dir_values[i]) for i in range(len(dir_values))])
        dir_values = [dir_values[i][:min_len] for i in range(len(dir_values))]
        mu_vals, std_vals = np.mean(dir_values, axis=0), np.std(dir_values, axis=0)
        # if len(min_values[f"{function}_{d_idx}_{d}"]) < len(num_directions):
        #   min_values[f"{function}_{d_idx}_{d}"].append(min_v)          
        # else:
        min_values[f"{function}_{d_idx}_{d}"] = min(min_values[f"{function}_{d_idx}_{d}"], min_v) if min_values[f"{function}_{d_idx}_{d}"] is not None else min_v
        values[f"{function}_{d_idx}_{d}"][dir_type].append((mu_vals, std_vals))  
      else:
        if f"{function}_{d_idx}_{d}" in init_values:
          del min_values[f"{function}_{d_idx}_{d}"]

          del init_values[f"{function}_{d_idx}_{d}"]

          del values[f"{function}_{d_idx}_{d}"]
        break
   #       print(f"MIN VALUE[{i}] for {l}", min_values[f"{function}_{d_idx}_{d}"][i], "for", dir_type )
#  print(min_values)
#  print(values, len(values[f"{function}_{d_idx}_{d}"]['spherical']))
#  exit()
#  print(directories[0])
#exit()
#for (problem_name, d, n) in problems:


print("[--] Results Loaded!")
taus_perf = np.logspace(-15, 0, 1000)


results = {dir_type : [[None for _ in range(taus_perf.shape[0])] for _ in range(len(num_directions)) ] for dir_type in direction_type}
for (j, tau) in enumerate(taus_perf):
  for (k, fun_key) in enumerate(min_values.keys()):
    for i in range(len(num_directions)):
      min_val = min_values[fun_key] 
      init_val = init_values[fun_key]
      for dir_type in direction_type:
        mu_values, _ = values[fun_key][dir_type][i]
        if np.any(np.isnan([np.min(mu_values), init_val,min_val])) or np.any(np.isinf([np.min(mu_values), init_val,min_val])):
          continue
        convergence_test = ((np.min(mu_values) - min_val + 1e-10) / (init_val - min_val + 1e-10)) <= tau
        if results[dir_type][i][j] is None:
          results[dir_type][i][j] = [convergence_test]
        else:
          results[dir_type][i][j].append(convergence_test)
          


taus = np.array([1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 1e0])
full_values = {dir_type : [[None for _ in range(taus.shape[0])] for _ in range(len(num_directions)) ] for dir_type in direction_type}
for (j, tau) in enumerate(taus):
  for (k, fun_key) in enumerate(min_values.keys()):
    for i in range(len(num_directions)):
      min_val = min_values[fun_key] #[i]
      init_val = init_values[fun_key]
      for dir_type in direction_type:
        mu_values, _ = values[fun_key][dir_type][i]
        if np.any(np.isnan([np.min(mu_values), init_val,min_val])) or np.any(np.isinf([np.min(mu_values), init_val,min_val])):
          continue
        full_convergence_test = ((mu_values - min_val + 1e-10) / (init_val - min_val + 1e-10))  <= tau
        if full_values[dir_type][i][j] is None:
          full_values[dir_type][i][j] = [full_convergence_test]
        else:
          full_values[dir_type][i][j].append(full_convergence_test)


# dir_type = 'spherical'
# for i in range(len(taus)):
#   print(dir_type, taus[i], [np.mean(ris) for ris in results[dir_type][i]])
# exit()
# for dir_type in direction_type:

#   print(dir_type, [np.mean(ris) for ris in results[dir_type]])

#print(results['coordinate'][-1][-1])
#exit()
print("[--] Plotting Fraction of Solved")

fig, axes = plt.subplots(1, len(num_directions), figsize=(20, 4))

#fig.suptitle("{} [$d = ${}]: {}".format(title, d, method.upper().replace("_","-")), fontsize=22)
axes = axes.flatten()


for i in range(len(num_directions)):
  ax = axes[i]
  ax.set_title("$\ell = $ {}".format(num_dirs_label[i]), fontsize=18)
  for (j, dir_type) in enumerate(direction_type):
    ax.plot(taus_perf, [np.mean(ris) for ris in results[dir_type][i]], '-', lw=3, label=dir_type.replace("_", " "), c=colors[j], rasterized=True)
  ax.set_xscale('log')
  ax.set_xlabel("$\\tau$", fontsize=18)
axes[0].legend()
axes[0].set_ylabel("$\\rho_{val}(\\tau)$", fontsize=18)
fig.savefig(f"./frac_solved_cutest.pdf", bbox_inches='tight')
plt.close(fig)


print("[++] Plotting Fraction Solved per Function Evaluations")

for tau_idx in [0, 1, 2, 3, 4, 5]:
  print("\t[--] Accuracy tau = ", taus[tau_idx])
  fig, axes = plt.subplots(1, len(num_directions), figsize=(20, 4))
  axes = axes.flatten()

  for i in range(len(num_directions)):
    ax = axes[i]
    ax.set_title("$\ell = $ {}".format(num_dirs_label[i]), fontsize=18)
    for (j, dir_type) in enumerate(direction_type):
#      print(full_values[dir_type][i][tau_idx])
      max_len = max([elem.shape[0] for elem in full_values[dir_type][i][tau_idx]])
      for k in range(len(full_values[dir_type][i][tau_idx])):
#        print(full_values[dir_type][i][tau_idx][k])
        if full_values[dir_type][i][tau_idx][k].shape[0] < max_len:
          full_values[dir_type][i][tau_idx][k] = np.pad(full_values[dir_type][i][tau_idx][k], (0, max_len - full_values[dir_type][i][tau_idx][k].shape[0]), mode='maximum' )
      vals = np.mean(full_values[dir_type][i][tau_idx], axis=0)
      ax.plot(range(1, len(vals) + 1), vals, '-', lw=3, label=dir_type.replace("_", " "), c=colors[j], rasterized=True)
#    ax.set_xscale('log')
    ax.set_xlabel("# function evaluations", fontsize=16)
    ax.set_ylim([-0.01, 1.01])
#    ax.set_xlim([1, 500])
#  axes[0].legend()
  axes[0].set_ylabel("$\\rho_{val}(\\tau)$", fontsize=18)
  fig.savefig(f"./frac_solved_fvals_profile_cutest_{taus[tau_idx]}.pdf", bbox_inches='tight')
  plt.close(fig)

