import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

input_dir = "./results/changing_l/ffd"
functions = ['LeastSquares', 'Rosenbrock', 'Rastrigin', 'StyblinskiTang', 'Trid', 'SumOfSquares']

dimensions = {
    8 : [2, 4, 6, 7, 8],
    20 : [2, 5, 10, 15, 18, 19, 20],
    50 : [2, 10, 25, 40, 48, 49, 50],
    100 : [2, 40, 50, 60, 80, 98, 99, 100],
    400 : [2, 50, 100, 150, 200, 300, 397, 398, 399, 400],
    500 : [100, 200, 250, 300, 400, 450, 498, 499, 500],
    1000 : [200, 300, 400, 450, 500, 550, 700, 800, 900, 998, 999, 1000],
}


color_map = {
    "spherical" : 'black',
    "householder" : 'darkgreen',
    "coordinate" : 'orange',
    "butterfly" : 'darkred',
    "butterfly_dct" : 'red',
    "qr" : 'blue',
    "optimized_spherical" : 'grey',
    "optimized_gaussian" : 'lightblue',
}
latex_formulas = {
    'LeastSquares': '$f(x) := \\frac{1}{2} \\| Ax - y \\|^2$',
    'Rosenbrock': '$f(x) := \\sum_{i=1}^{d-1} \\left[ (1.0 - x_i)^2 + 100.0 \\cdot (x_{i+1} - x_i^2)^2 \\right]$',
    'Rastrigin': '$f(x) := 10d + \\sum_{i=1}^{d} \\left[ x_i^2 - 10 \\cos(2\\pi x_i) \\right]$',
    'StyblinskiTang': '$f(x) := \\frac{1}{2} \\sum_{i=1}^{d} (x_i^4 - 16x_i^2 + 5x_i)$',
    'Trid': '$f(x) := \\sum_{i=1}^{d} (x_i - 1)^2 - \\sum_{i=2}^{d} x_{i-1}x_i$',
    'SumOfSquares': '$f(x) := \\sum_{i=1}^{d} x_i^2$',
}


def read_results(input_dirs, fun_name, d, l_list):
    results = {}
    elem1, elem2 = 3, 4
    for input_dir in input_dirs:
        for l in l_list:
            with open("{}/{}_{}_{}.log".format(input_dir, fun_name, d, l), 'r') as f:
                for line in f.readlines():
                    splitted = line.split(",")
                    if splitted[0] not in results:
                        results[splitted[0]] = ([float(splitted[elem1])], [float(splitted[elem2])])
                    else:
                        results[splitted[0]][0].append(float(splitted[elem1]))
                        results[splitted[0]][1].append(float(splitted[elem2]))            
    return results
#paths = [input_dir, './results/changing_l/optimized']
paths = [input_dir]
for (d, l_list) in dimensions.items():
    print(f"[--] Plotting results for $d = {d}$")
    result_per_functions = {}
    for fun_name in functions:
        
        result_per_functions[fun_name] =read_results(paths, fun_name, d, l_list)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Approximation error ($d = {}$)".format(d), fontsize=20)
    line1, line2 = axes[0], axes[1]
    for i in range(len(functions) // 2):
        result_fun = result_per_functions[functions[i]]    
        for (label, values) in result_fun.items():
            mu, std = values[0], values[1]
            mu = np.array(mu)
            std = np.array(std)
            lb, ub = mu - std, mu + std
            if len(mu) != len(dimensions[d]):
                continue
#            print(label, fun_name, len(mu), len(dimensions[d]))
            if label in color_map:
                line1[i].plot(dimensions[d], mu, 'o-', c=color_map[label], label=label.replace("_"," "))
                line1[i].fill_between(dimensions[d], lb, ub, color=color_map[label], alpha=0.6)
        line1[i].set_title(functions[i], fontsize=14)
        line1[i].set_yscale('log')
        line1[i].set_xscale('log')
        if i == 0:
            line1[i].legend(fontsize=12)
        line1[i].set_xlabel("$d$", fontsize=14)
        line1[i].set_ylabel("Time ($s$)", fontsize=14)

    for i in range(len(functions) // 2, len(functions)):
        result_fun = result_per_functions[functions[i]]   
        for (label, values) in result_fun.items():
            mu, std = values[0], values[1]
            mu = np.array(mu)
            std = np.array(std)
            lb, ub = mu - std, mu + std
            if label in color_map:
                line2[i - 3].plot(dimensions[d], mu, 'o-', c=color_map[label], label=label.replace("_"," "))
                line2[i - 3].fill_between(dimensions[d], lb, ub, color=color_map[label], alpha=0.6)
        line2[i - 3].set_title(functions[i], fontsize=14)
        line2[i - 3].set_yscale('log')
        line2[i - 3].set_xscale('log')
        line2[i - 3].set_xlabel("$d$", fontsize=14)
        line2[i - 3].set_ylabel("Time ($s$)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{input_dir}/estimator_quality_l_{d}_time.pdf", bbox_inches='tight')
    plt.close(fig)
    #print(line1)
    #print(result_per_functions)