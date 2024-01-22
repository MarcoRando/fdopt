import numpy as np 

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
dimensions = [20, 50, 100, 500, 1000, 2000]
num_directions = [lambda d : 1, lambda d : d // 4, lambda d : d // 2, lambda d : d - 1]

def read_results(input_dir, d, num_directions):
    results = {}
    for num_dir in num_directions:
        l = num_dir(d)
        with open("{}/time_costs_{}_{}.log".format(input_dir, d, l), 'r') as f:
            for line in f.readlines():
                splitted = line.split(",")
                if splitted[0] not in results:
                    results[splitted[0]] = ([float(splitted[1])], [float(splitted[2])])
                else:
                    results[splitted[0]][0].append(float(splitted[1]))
                    results[splitted[0]][1].append(float(splitted[2]))
    return results


for d in dimensions:
    results = read_results("./results/time_cost_dhalf", d, num_directions)
    dirs = [num_dir(d) for num_dir in num_directions]
    fig, ax = plt.subplots(figsize=(8,6))

    for (label, values) in results.items():
        mu, std = values[0], values[1]
        mu = np.array(mu)
        std = np.array(std)
        lb, ub = mu - std, mu + std
        ax.plot(dirs, mu, 'o-', label=label)
        ax.fill_between(dirs, abs(lb), ub, alpha=0.6)
        
    ax.set_xlabel("$\ell$", fontsize=18)
    ax.set_ylabel("Computational Time ($s$)", fontsize=18)
    ax.set_yscale("log")
    ax.set_title("Sampling cost of {} x $\ell$ matrices".format(d), fontsize=20)

    ax.legend(fontsize=18)
    fig.savefig("./results/time_cost_dhalf/time_costs_{}_log.pdf".format(d), bbox_inches='tight')
    
