import numpy as np
import matplotlib.pyplot as plt

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


color_map = {
    "spherical" : 'black',
    "householder" : 'darkgreen',
    "coordinate" : 'orange',
    "butterfly" : 'darkred',
    "butterfly_dct" : 'red',
    "qr" : 'blue',
}

def read_results(input_dir, fun_name, d):
    results = {}
    with open("{}/{}_{}.log".format(input_dir, fun_name, d), 'r') as f:
        for line in f.readlines():
            splitted = line.split(",")
            if splitted[0] not in results:
                results[splitted[0]] = ([float(splitted[1])], [float(splitted[2])])
            else:
                results[splitted[0]][0].append(float(splitted[1]))
                results[splitted[0]][1].append(float(splitted[2]))
    return results
infile = "./results/ffd"
fname = "LeastSquares_1000"
d=  1000
results = read_results(infile, fname, d)


fig, ax = plt.subplots()
ax.set_title("Least Squares", fontsize=16)
i = 0
means, stds, labels, colors = [], [], [], []
for (label, (mu, std)) in results.items():
  #  mu, std = np.array([mu]), np.array([std])
    means.append(mu)
    stds.append([abs(np.array(mu) - np.array(std)), np.array(mu) + np.array(std)])
    labels.append(label)    
    colors.append(color_map[label])
    ax.plot(i, mu, 'o', color=color_map[label], label=label)
    ax.fill_between(i, abs(np.array(mu) - np.array(std)), np.array(mu) + np.array(std), alpha=0.6, color=color_map[label])
    i+=1
#boxplot =ax.boxplot(means, conf_intervals=stds, labels=labels, patch_artist=True)

# for box, color in zip(boxplot['boxes'], colors):
#     box.set_facecolor(color)
#ax.set_yscale("log")
#ax.legend()
ax.set_xticks(range(len(labels)), labels)
ax.set_ylabel("Number of function evaluations", fontsize=14)
ax.set_xlabel("Direction Type", fontsize=14)
fig.tight_layout()
fig.savefig("./{}_{}.pdf".format(fname, d), bbox_inches='tight')
    