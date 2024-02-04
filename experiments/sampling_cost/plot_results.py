import numpy as np 

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
dimensions = [2, 10, 12, 15, 20, 50, 100, 150, 200, 500, 1000]#, 2000, 5000, 8000, 10000]

color_map = {
    "gaussian" : 'lightgreen',
    "spherical" : 'black',
    "householder" : 'darkgreen',
    "coordinate" : 'orange',
    "butterfly" : 'darkred',
    "butterfly_dct" : 'red',
    "qr" : 'blue',
}


def read_results(input_dir, dimensions):
    results = {}
    for d in dimensions:
        with open("{}/time_costs_{}.log".format(input_dir, d), 'r') as f:
            for line in f.readlines():
                splitted = line.split(",")
                if splitted[0] not in results:
                    results[splitted[0]] = ([float(splitted[1])], [float(splitted[2])])
                else:
                    results[splitted[0]][0].append(float(splitted[1]))
                    results[splitted[0]][1].append(float(splitted[2]))
    return results

results = read_results("./results/time_cost", dimensions)

fig, ax = plt.subplots(figsize=(8,6))

for (label, values) in results.items():
    mu, std = values[0], values[1]
    mu = np.array(mu)
    std = np.array(std)
    lb, ub = mu - std, mu + std
#    if label=='qr' or label=='spherical':
    ax.plot(dimensions, mu, 'o-',c = color_map[label], label=label.replace("_", " "))
    ax.fill_between(dimensions, abs(lb), ub, color = color_map[label], alpha=0.6)
    
ax.set_xlabel("$d$", fontsize=18)
ax.set_ylabel("Computational Time ($s$)", fontsize=18)

ax.legend(fontsize=18)
ax.set_yscale("log")
ax.set_xscale("log")
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
