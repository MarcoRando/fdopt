import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

input_dir = "./results/changing_d/cfd"
functions = ['LeastSquares', 'Rosenbrock', 'Rastrigin', 'StyblinskiTang', 'Trid', 'SumOfSquares']
dimensions = [2, 4, 8, 10, 20, 50, 100, 200, 400, 500, 1000]
color_map = {
    "spherical" : 'black',
    "householder" : 'darkgreen',
    "coordinate" : 'orange',
    "butterfly" : 'darkred',
    "butterfly_dct" : 'red',
    "qr" : 'blue',
}

def read_results(input_dir, fun_name, dimensions):
    results = {}
    for d in dimensions:
        with open("{}/{}_{}.log".format(input_dir, fun_name, d), 'r') as f:
            for line in f.readlines():
                splitted = line.split(",")
                if splitted[0] not in results:
                    results[splitted[0]] = ([float(splitted[1])], [float(splitted[2])])
                else:
                    results[splitted[0]][0].append(float(splitted[1]))
                    results[splitted[0]][1].append(float(splitted[2]))
    return results
result_per_functions = {}
for fun_name in functions:
    result_per_functions[fun_name] =read_results(input_dir, fun_name, dimensions)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))

line1, line2 = axes[0], axes[1]
for i in range(len(functions) // 2):
    result_fun = result_per_functions[functions[i]]    
    for (label, values) in result_fun.items():
        mu, std = values[0], values[1]
        mu = np.array(mu)
        std = np.array(std)
        lb, ub = mu - std, mu + std
        line1[i].plot(dimensions, mu, 'o-', c=color_map[label], label=label.replace("_"," "))
        line1[i].fill_between(dimensions, lb, ub, color=color_map[label], alpha=0.6)
    line1[i].set_title(functions[i], fontsize=14)
    line1[i].set_yscale('log')
    line1[i].set_xscale('log')
    if i == 0:
        line1[i].legend(fontsize=12)
    line1[i].set_xlabel("$d$", fontsize=14)
    line1[i].set_ylabel("Approximation Error", fontsize=14)

for i in range(len(functions) // 2, len(functions)):
    result_fun = result_per_functions[functions[i]]   
    for (label, values) in result_fun.items():
        mu, std = values[0], values[1]
        mu = np.array(mu)
        std = np.array(std)
        lb, ub = mu - std, mu + std
        line2[i - 3].plot(dimensions, mu, 'o-', c=color_map[label], label=label.replace("_"," "))
        line2[i - 3].fill_between(dimensions, lb, ub, color=color_map[label], alpha=0.6)
    line2[i - 3].set_title(functions[i], fontsize=14)
    line2[i - 3].set_yscale('log')
    line2[i - 3].set_xscale('log')
    line2[i - 3].set_xlabel("$d$", fontsize=14)
    line2[i - 3].set_ylabel("Approximation Error", fontsize=14)
fig.tight_layout()
fig.savefig(f"{input_dir}/estimator_quality_d.pdf", bbox_inches='tight')
#print(line1)
#print(result_per_functions)