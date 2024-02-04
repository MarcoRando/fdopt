import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

input_dir = ["./results/changing_d/ffd"]
functions = ['LeastSquares', 'Rosenbrock', 'Rastrigin', 'StyblinskiTang', 'Trid', 'SumOfSquares']
dimensions = [2, 4, 8, 10, 20, 50, 100, 200, 400, 500, 1000]
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
fun_names_p = {
    'LeastSquares' : "Least Squares", 
    'Rosenbrock' : 'Rosenbrock', 
    'Rastrigin' : 'Rastrigin', 
    'StyblinskiTang': 'Styblinski-Tang', 
    'Trid' : 'Trid', 
    'SumOfSquares' : 'Sum of Squares'
}
def read_results(input_dirs, fun_name, d, results):
#    results = {}
#    elem1, elem2 = 1, 2
    for input_dir in input_dirs:
        with open("{}/{}_{}.log".format(input_dir, fun_name, d), 'r') as f:
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
result_per_functions = {}
#paths = [input_dir]#, './results/changing_d/optimized']

for fun_name in functions:
    result_per_functions[fun_name] = {}
    for d in dimensions:
        _ =read_results(input_dir, fun_name, d, result_per_functions[fun_name])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("{}".format(fun_names_p[fun_name]), fontsize=20)

    for (label, values) in result_per_functions[fun_name].items():
            mu, std = values[0], values[1]
            mu_t, std_t = values[2], values[3]
            mu, std = np.array(mu), np.array(std)
            mu_t, std_t = np.array(mu_t), np.array(std_t)

            lb, ub = mu - std, mu + std
            lb_t, ub_t = mu_t - std_t, mu_t + std_t
            if len(mu) != len(dimensions):
                continue
            if label in color_map:
                ax1.plot(dimensions, mu, 'o-', c=color_map[label], label=label.replace("_"," "))
                ax1.fill_between(dimensions, lb, ub, color=color_map[label], alpha=0.6)
                ax2.plot(dimensions, mu_t, 'o-', c=color_map[label], label=label.replace("_"," "))
                ax2.fill_between(dimensions, lb_t, ub_t, color=color_map[label], alpha=0.6)
            ax1.set_yscale('log')
            ax1.set_xscale('log')
            ax1.legend(fontsize=12)
            ax1.set_title("Gradient Approximation Error", fontsize=16)
            ax1.set_xlabel("$d$", fontsize=14)
            ax1.set_ylabel("Approximation Error", fontsize=14)
            ax2.set_title("Computational Time", fontsize=16)
            ax2.set_yscale('log')
            ax2.set_xscale('log')
            ax2.set_xlabel("$\ell$", fontsize=14)
            ax2.set_ylabel("Time ($s$)", fontsize=14)


    fig.tight_layout()
    fig.savefig(f"{input_dir[0]}/{fun_name}_estimator_quality_d.pdf", bbox_inches='tight')
print(result_per_functions['Rastrigin'])

#print(line1)
#print(result_per_functions)