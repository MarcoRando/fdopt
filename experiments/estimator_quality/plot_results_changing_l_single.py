import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

input_dir = "./results/changing_l/cfd"
functions = ['LeastSquares', 'Rosenbrock', 'Rastrigin', 'StyblinskiTang', 'Trid', 'SumOfSquares']

fun_names_p = {
    'LeastSquares' : "Least Squares", 
    'Rosenbrock' : 'Rosenbrock', 
    'Rastrigin' : 'Rastrigin', 
    'StyblinskiTang': 'Styblinski-Tang', 
    'Trid' : 'Trid', 
    'SumOfSquares' : 'Sum of Squares'
}

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
#    elem1, elem2 = 1, 2
    for input_dir in input_dirs:
        for l in l_list:
            with open("{}/{}_{}_{}.log".format(input_dir, fun_name, d, l), 'r') as f:
                for line in f.readlines():
                    splitted = line.split(",")
                    if splitted[0] not in results:
                        results[splitted[0]] = ([float(splitted[1])], [float(splitted[2])], [float(splitted[3])], [float(splitted[4])])
                    else:
                        results[splitted[0]][0].append(float(splitted[1]))
                        results[splitted[0]][1].append(float(splitted[2]))            
                        results[splitted[0]][2].append(float(splitted[3]))
                        results[splitted[0]][3].append(float(splitted[4]))            

    return results
paths = [input_dir, './results/changing_l/optimized']
#paths = [input_dir]
for (d, l_list) in dimensions.items():
    print(f"[--] Plotting results for $d = {d}$")
    result_per_functions = {}
    for fun_name in functions:
        
        result_per_functions[fun_name] =read_results(paths, fun_name, d, l_list)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("{} ($d = {}$)".format(fun_names_p[fun_name], d), fontsize=20)
        for (label, values) in result_per_functions[fun_name].items():
            mu, std = values[0], values[1]
            mu_t, std_t = values[2], values[3]
            mu, std = np.array(mu), np.array(std)
            mu_t, std_t = np.array(mu_t), np.array(std_t)

            lb, ub = mu - std, mu + std
            lb_t, ub_t = mu_t - std_t, mu_t + std_t
            if len(mu) != len(dimensions[d]):
                continue
            if label in color_map:
                ax1.plot(dimensions[d], mu, 'o-', c=color_map[label], label=label.replace("_"," "))
                ax1.fill_between(dimensions[d], lb, ub, color=color_map[label], alpha=0.6)
                ax2.plot(dimensions[d], mu_t, 'o-', c=color_map[label], label=label.replace("_"," "))
                ax2.fill_between(dimensions[d], lb_t, ub_t, color=color_map[label], alpha=0.6)
            ax1.set_yscale('log')
            ax1.set_xscale('log')
            ax1.legend(fontsize=12)
            ax1.set_title("Gradient Approximation Error", fontsize=16)
            ax1.set_xlabel("$\ell$", fontsize=14)
            ax1.set_ylabel("Approximation Error", fontsize=14)
            ax2.set_title("Computational Time", fontsize=16)
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.set_xlabel("$\ell$", fontsize=14)
            ax2.set_ylabel("Time ($s$)", fontsize=14)

        fig.tight_layout()
        fig.savefig(f"{input_dir}/{fun_name}_estimator_quality_l_{d}_single.pdf", bbox_inches='tight')
        plt.close(fig)
