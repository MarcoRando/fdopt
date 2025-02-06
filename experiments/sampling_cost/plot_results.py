import numpy as np 
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
dimensions = [1, 10, 50, 100, 500, 1000, 5000, 10000]# [1, 2, 10, 50, 100, 200, 500, 1000, 2000, 5000, 8000, 10000]

color_map = {
    "gaussian" : 'lightgreen',
    "spherical" : 'black',
    "householder" : 'darkgreen',
    "coordinate" : 'orange',
    "butterfly" : 'brown',
    "butterfly_dct" : 'red',
    "qr" : 'blue',
    "svd" : 'purple',
}

devices = ['cpu', 'cuda']
dtypes = [torch.float32, torch.float64]


def read_results(input_dir, dimensions):
    results = {}
    for device in devices:
        results[device] = {
            dtypes[0] : {},
            dtypes[1] : {}
        }
    for device in devices:
        for dtype in dtypes:                
            for d in dimensions:
                with open("{}/time_costs_{}_{}_{}.log".format(input_dir, d, device, dtype), 'r') as f:
                    for line in f.readlines():
                        splitted = line.split(",")
                        if splitted[0] not in results[device][dtype]:
                            results[device][dtype][splitted[0]] = ([float(splitted[1])], [float(splitted[2])])
                        else:
                            results[device][dtype][splitted[0]][0].append(float(splitted[1]))
                            results[device][dtype][splitted[0]][1].append(float(splitted[2]))
    return results

results = read_results("./results/time_cost", dimensions)


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

ris_0_0 = results[devices[0]][dtypes[0]]
ris_0_1 = results[devices[0]][dtypes[1]]
ris_1_0 = results[devices[1]][dtypes[0]]
ris_1_1 = results[devices[1]][dtypes[1]]

for (label, values) in ris_0_0.items():
    mu, std = values[0], values[1]
    mu, std = np.array(mu), np.array(std)
    lb, ub = mu - std, mu + std
    axes[0].plot(dimensions, mu, 'o-',c = color_map[label], label=label.replace("_", " "))
    axes[0].fill_between(dimensions, abs(lb), ub, color = color_map[label], alpha=0.6)

# for (label, values) in ris_0_1.items():
#     mu, std = values[0], values[1]
#     mu, std = np.array(mu), np.array(std)
#     lb, ub = mu - std, mu + std
#     axes[1][0].plot(dimensions, mu, 'o-',c = color_map[label], label=label.replace("_", " "))
#     axes[1][0].fill_between(dimensions, abs(lb), ub, color = color_map[label], alpha=0.6)

# for (label, values) in ris_1_1.items():
#     mu, std = values[0], values[1]
#     mu, std = np.array(mu), np.array(std)
#     lb, ub = mu - std, mu + std
#     axes[1][1].plot(dimensions, mu, 'o-',c = color_map[label], label=label.replace("_", " "))
#     axes[1][1].fill_between(dimensions, abs(lb), ub, color = color_map[label], alpha=0.6)


for (label, values) in ris_1_0.items():
    mu, std = values[0], values[1]
    mu, std = np.array(mu), np.array(std)
    lb, ub = mu - std, mu + std
    axes[1].plot(dimensions, mu, 'o-',c = color_map[label], label=label.replace("_", " "))
    axes[1].fill_between(dimensions, abs(lb), ub, color = color_map[label], alpha=0.6)

    
fig.supylabel("Computational Time ($s$)", fontsize=18)

axes[0].set_title("CPU", fontsize=18)
axes[1].set_title("GPU", fontsize=18)

for i in range(1):
    for j in range(2):
        axes[j].set_xlabel("$d$", fontsize=18)
        axes[j].set_xscale("log")
        axes[j].set_yscale("log")

axes[0].legend(fontsize=14)
fig.tight_layout()
fig.savefig("./time_costs_log_log.pdf", bbox_inches='tight')
    
# dimensions = [1, 2, 10, 12, 15, 20, 50, 100, 150, 200, 500, 1000, 2000, 5000]
# results = read_results("./results/time_cost_bt", dimensions)

# fig, ax = plt.subplots(figsize=(8,6))

# for (label, values) in results.items():
#     mu, std = values[0], values[1]
#     mu = np.array(mu)
#     std = np.array(std)
#     lb, ub = mu - std, mu + std
#     ax.plot(dimensions, mu, 'o-', c = color_map[label], label=label.replace("_", " "))
#     ax.fill_between(dimensions, abs(lb), ub, color = color_map[label], alpha=0.6)
    
# ax.set_xlabel("$d$", fontsize=18)
# ax.set_ylabel("Computational Time ($s$)", fontsize=18)

# ax.legend(fontsize=18)
# fig.savefig("./time_costs_bt_log.pdf", bbox_inches='tight')
