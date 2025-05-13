import numpy as np 
import matplotlib.pyplot as plt
import sys
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

results_path = sys.argv[1]
nreps = int(sys.argv[2])

d = 784
num_directions =  [2, d//3, d//2, round((4/5) * d),  d]
num_dirs_label = ["2", "d/3", "d/2",  "4/5 d",  'd']


direction_types = ['spherical'  , 'gaussian', 'rademacher', 'coordinate', 'householder', 'butterfly', 'permuted_householder', 'qr']
colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']

def load_fvalues():
    num_evals = {dir_type : [[] for _ in range(len(num_directions))] for dir_type in direction_types}
    results = {dir_type : [[] for _ in range(len(num_directions))] for dir_type in direction_types}
    min_values = []#  [[] for _ in range(len(num_directions))] 
    init_values = []# [[] for _ in range(len(num_directions))] 
    for i in range(len(num_directions)):
        l = num_directions[i]
        for dir_type in direction_types:
            #im_id = 0
            mu_values, n_evals = [], []
            print(f"[--] Loading results for {dir_type} [l = {l}]")
            with open(f"{results_path}/adv_exp/mnist/fun_values/fun_values_mnist_{dir_type}_{l}.log", 'r') as f:
                current_min = None
                prev_img = 0
                prev_rep = 0
                for line in f.readlines():
                    splitted = line.split(",")
                    im_id = int(splitted[0])
                    f0 = float(splitted[1])
                    fk = float(splitted[2])
                    n_eval = int(splitted[3])
                    n_rep = int(splitted[4])
                    current_min = min(fk, current_min) if current_min is not None else fk
#                    if len(init_values[i]) <= im_id:
#                        init_values[i].append(f0)
                    if prev_img is None or im_id != prev_img:#int(splitted[0]):
#                        norm_results[dir_type][i].append(np.mean(mu_norm_distortion))
#                        results[dir_type][i].append(np.mean(mu_distortion))
                   #     if i == 0 and dir_type == 'spherical':
                   #         print("Adding", len(mu_values))
                        mu_values = mu_values[ : nreps*29000]
                        results[dir_type][i].append(mu_values)
                        num_evals[dir_type][i].append(n_evals)
                        print(dir_type, len(results[dir_type][i][-1]), len(mu_values), im_id)
                        #exit()
#                        if len(init_values[i]) <= im_id:
#                            init_values[i].append(f0)
#                        init_values[i][im_id] = f0
                        if len(min_values)  > im_id:
                            min_values[im_id] = current_min if current_min < min_values[im_id] else min_values[im_id]
                            init_values[im_id] = f0
                        else:
                            init_values.append(f0)
                            min_values.append(current_min)
                        current_min = None
                    #    im_id += 1
                        mu_values, n_evals = [], []
                        prev_img = im_id
                        prev_rep = 0
                    if n_rep != prev_rep:
                        mu_values = mu_values[ : (prev_rep + 1)*29000]
                    mu_values += [fk for _ in range(n_eval)] #.append(fk)
                    #print(prev_rep, n_rep, len(mu_values))
                    n_evals.append(n_eval)
                    prev_rep = n_rep
#                    normalized_distortion = float(splitted[2])
                    # mu_norm_distortion.append(normalized_distortion)
                    # mu_distortion.append(float(splitted[1]))
#                    results[dir_type][i].append(float(splitted[1]))
    return results, min_values, init_values, num_evals





taus = np.logspace(-1, 0, 100)
results, min_values, init_values, num_evals = load_fvalues()

print("[--] Results Loaded!")

# STORED AS -> results[direction][num_direction][image]

# perf_profile = []
# fig, axes = plt.subplots(1, len(num_directions), figsize=(20,4))
# axes = axes.flatten()
# for num_direction_idx in range(len(num_directions)):
#     print("[--] Plotting l = ", num_dirs_label[num_direction_idx])
#     ax = axes[num_direction_idx]
#     ax.set_title(f"$\\ell = {num_dirs_label[num_direction_idx]}$", fontsize=18)
#     for (dir_idx, dir_type) in enumerate(direction_types):
#         profile = []
#         for (tau_idx, tau) in enumerate(taus):
#             perf_met = [] # [np.mean((np.array(results[dir_type][num_direction_idx][j]).reshape(nreps, -1)[:, -1] - min_values[j]) / (init_values[num_direction_idx][j] - min_values[num_direction_idx][j]), axis=0) <= tau for j in range(len(results[dir_type][num_direction_idx]))]
#             for j in range(len(results[dir_type][num_direction_idx])):
#                 init_diff = init_values[j] - min_values[j]
#                 value_diff = np.array(results[dir_type][num_direction_idx][j]).reshape(nreps, -1)[:, -1] - min_values[j]
#                 mu_vals = np.clip(value_diff / init_diff, 0.0, 1e10)
# #                print(mu_vals)
#                 perf_met.append(np.mean(mu_vals))
#                 print("[{}/{}] tau = {}/{} perf_met = {}".format(j, len(results[dir_type][num_direction_idx]), tau_idx, len(taus), perf_met[-1]))
#                 assert perf_met[-1] >= 0.0
#             perf_met = np.mean(np.mean(perf_met, axis=0) <= tau)
# #            perf_met = np.mean(perf_met)
#             profile.append(perf_met)
#         ax.plot(taus, profile, '-', lw=3, label=dir_type.replace("_", " "), c=colors[dir_idx], rasterized=True)
#     #    perf_profile.append(profile)
#     #    print(f"[{dir_type}] {perf_met}")
#     ax.set_xscale("log")
#     ax.set_xlabel("$\\tau$", fontsize=18)
#     if num_direction_idx == 0:
#         ax.set_ylabel("$\\rho_{val}(\\tau)$", fontsize=18)
#         ax.legend(loc='upper left', fontsize=10)
# fig.savefig("./fraction_solved.pdf", bbox_inches='tight')
# plt.close(fig)

print("[--] Plotted performance profiles")

tau = 9e-1

# results[direction][num_direction][image]
#perf_profile = []

for tau in [0.1, 0.5, 1.0]:
    fig, axes = plt.subplots(1, len(num_directions), figsize=(20,4))
    axes = axes.flatten()
    for num_direction_idx in range(len(num_directions)):
        ax = axes[num_direction_idx]
        ax.set_title(f"$\\ell = {num_dirs_label[num_direction_idx]}$", fontsize=18)
        for (dir_idx, dir_type) in enumerate(direction_types):
            profile = []
            print("[--] Solving for ", dir_type)
            for j in range(len(num_evals[dir_type][num_direction_idx])):
                ris = np.array(results[dir_type][num_direction_idx][j]).reshape(nreps, -1)
    #            ris = np.mean(ris, axis=0)
                conv_test_single_img = np.mean( (ris - min_values[j]) / (init_values[j] - min_values[j]) <= tau, axis=0)
                profile.append(conv_test_single_img)
            # min_len = min([len(elem) for elem in profile])
            # for k in range(len(profile)):
            #     if len(profile[k]) > 10000:    
            #         profile[k] = profile[k][:10000] #np.pad(profile[k], (0, max_len - len(profile[k])), mode='minimum' )
            # print([len(profile[k]) for k in range(len(profile))])
            profile = np.mean(profile, axis=0)

            ax.plot(range(len(profile)), profile, '-', lw=3, label=dir_type.replace("_", " "), c=colors[dir_idx], rasterized=True)
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel("Function Evaluations", fontsize=18)
        if num_direction_idx == 0:
            ax.set_ylabel("$\\rho_{val}(\\tau)$", fontsize=18)
            ax.legend(loc='upper left', fontsize=10)
    fig.savefig(f"./frac_solved_fvals_{tau}.pdf", bbox_inches='tight')
    plt.close(fig)

