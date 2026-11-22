import numpy as np
import sys
import matplotlib
from matplotlib.lines import Line2D

# same colours, labels, family order and matplotlib style as the other plotting scripts
from plot_common import COLOR, LABEL, ORDER, l_label, plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

LABEL_FS = 16      # axis labels (same as the other figures)
LW = 2             # line width used in the other figures

results_path = sys.argv[1]
nreps = int(sys.argv[2])

d = 784
num_directions =  [2, d//3, d//2, round((4/5) * d),  d]
num_dirs_label = ["2", "d/3", "d/2",  "4/5 d",  'd']


direction_types = ['spherical'  , 'gaussian', 'rademacher', 'coordinate', 'householder', 'butterfly', 'permuted_householder', 'qr']

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


results, min_values, init_values, num_evals = load_fvalues()

print("[--] Results Loaded!")

# STORED AS -> results[direction][num_direction][image]

# same family order as in the other figures
plot_types = [f for f in ORDER if f in direction_types]

for tau in [0.1, 0.5, 1.0]:
    fig, axes = plt.subplots(1, len(num_directions), figsize=(20, 4.6))
    axes = axes.flatten()
    handles = {}
    for num_direction_idx in range(len(num_directions)):
        ax = axes[num_direction_idx]
        ax.set_title(l_label(num_directions[num_direction_idx], d))
        for dir_type in plot_types:
            profile = []
            print("[--] Solving for ", dir_type)
            for j in range(len(num_evals[dir_type][num_direction_idx])):
                ris = np.array(results[dir_type][num_direction_idx][j]).reshape(nreps, -1)
    #            ris = np.mean(ris, axis=0)
                conv_test_single_img = np.mean( (ris - min_values[j]) / (init_values[j] - min_values[j]) <= tau, axis=0)
                profile.append(conv_test_single_img)
            profile = np.mean(profile, axis=0)

            h, = ax.plot(range(len(profile)), profile, '-', lw=LW, c=COLOR[dir_type], rasterized=True)
            handles.setdefault(dir_type, h)
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel("# function evaluations", fontsize=LABEL_FS)
        if num_direction_idx == 0:
            ax.set_ylabel("$\\rho_{val}(\\tau)$", fontsize=LABEL_FS)
    # legend below the panels, 5 per row, as in the other figures
    fig.legend([handles[f] for f in plot_types], [LABEL[f] for f in plot_types], loc="lower center",
               ncol=min(len(plot_types), 5), frameon=False, handlelength=2.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(f"./frac_solved_fvals_{tau}.pdf", bbox_inches='tight')
    plt.close(fig)