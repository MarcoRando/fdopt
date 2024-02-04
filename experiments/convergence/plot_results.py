import numpy as np
import matplotlib.pyplot as plt

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


color_map = {
    "gaussian" : 'lightgreen',
    "spherical" : 'black',
    "householder" : 'darkgreen',
    "coordinate" : 'orange',
    "butterfly" : 'darkred',
    "butterfly_dct" : 'red',
    "qr" : 'blue',
    "optimized_spherical" : 'grey',
    "optimized_gaussian" : 'lightblue',
}

def read_results(input_dir, fun_name, d):
    results = {}
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

infile = "./results/ffd"
d=  500
fname = "LeastSquares_{}".format(d)
results = read_results(infile, fname, d)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Least Squares (d = {})".format(d), fontsize=18)
ax1.set_title("Convergence", fontsize=16)
ax2.set_title("Computational Time", fontsize=16)
i = 0
means, stds, labels, colors = [], [], [], []
means_tm, stds_tm = [], []
for (label, (mu, std, mu_t, std_t)) in results.items():
  #  mu, std = np.array([mu]), np.array([std])
    # if label == 'gaussian':
    #     continue

    means.append(mu)
    stds.append([abs(np.array(mu) - np.array(std)), np.array(mu) + np.array(std)])
    means_tm.append(mu_t)
    stds_tm.append([abs(np.array(mu_t) - np.array(std_t)), np.array(mu_t) + np.array(std_t)])
    labels.append(label)    
    colors.append(color_map[label])
    ax1.plot(i, mu, 'o', color=color_map[label], label=label.replace("_", " "))
    ax1.fill_between(i, abs(np.array(mu) - np.array(std)), np.array(mu) + np.array(std), alpha=0.6, color=color_map[label])
    ax2.plot(i, mu_t, 'o', color=color_map[label], label=label.replace("_", " "))
    ax2.fill_between(i, abs(np.array(mu_t) - np.array(std_t)), np.array(mu_t) + np.array(std_t), alpha=0.6, color=color_map[label])
    i+=1
ax1.set_yscale("log")
ax2.set_yscale("log")
#ax.legend()
ax1.set_xticks(range(len(labels)), [lab.replace("_", " ") for lab in labels], rotation=90)
ax1.set_ylabel("Number of function evaluations", fontsize=14)
ax1.set_xlabel("Direction Type", fontsize=14)
ax2.set_xticks(range(len(labels)), [lab.replace("_", " ") for lab in labels], rotation=90)
ax2.set_ylabel("Time ($s$)", fontsize=14)
ax2.set_xlabel("Direction Type", fontsize=14)

fig.tight_layout()
fig.savefig("./{}_{}.pdf".format(fname, d), bbox_inches='tight')
    