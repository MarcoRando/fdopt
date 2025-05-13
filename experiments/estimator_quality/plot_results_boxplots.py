import sys
import numpy as np 

import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
direction_type = ['spherical', 'gaussian', 'rademacher', 'coordinate', 'householder', 'butterfly', 'permuted_householder', 'qr']
colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown', 'pink', 'gray']

fun_name = sys.argv[1]
d = int(sys.argv[2])
base_path = sys.argv[3]

if fun_name == 'logistic':
    fname = "LogisticReg_1000"
    title = "Logistic Regression"
elif fun_name == 'least_squares':
    fname = "LeastSquares_10000.0_1.0"
    title = "Least Squares"
elif fun_name == 'trid':
    fname = title = "Trid"
elif fun_name == 'sparse_sum':
    fname = title = "Sparse Sum"
    s = int(sys.argv[4])
    fname = f"SparseSum_{s}"
elif fun_name == 'quing':
    fname = "Quing"
    title = "Qing"
elif fun_name == 'rosenbrock':
    fname = title = "Rosenbrock"
elif fun_name == 'stib':
    fname = title = "StyblinskiTang"



#d = int(sys.argv[3]) #dimensions[0]

num_directions = [2, d//3, d//2, round(d * (4/5)), d] #list(range(1, d, 2)) + [d]
l_labels = ['2', 'd/3', 'd/2', '4/5 d', 'd']

fig, ax = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle(title, fontsize=20)
data = pd.read_csv(f"{base_path}/grad_approx_error/noise_0.0/ffd/{fname}_{d}_1e-07.csv")

print(f"[++] Dimension d = {d}")

for (i, l) in enumerate(num_directions):
    ax[i].set_title(f"$\ell = {l_labels[i]}$", fontsize=16)  
    data_per_direction = data[data['l'] == l]

    errors = []
    for dir_type in direction_type:
        data_dir = data_per_direction[data_per_direction['directions']==dir_type]          
        errors.append(data_dir['err'].values)
    b = ax[i].boxplot(errors, positions=np.arange(len(direction_type)), patch_artist=True)
    ax[i].set_yscale("log")
    ax[i].set_xticks(np.arange(len(direction_type)), [dir_type[0].upper() for dir_type in direction_type])
    for j in range(len(b['boxes'])):
        b['boxes'][j].set_color(colors[j])
    for j in range(len(b['medians'])):
        b['medians'][j].set_color('black')

fig.supylabel("$\\frac{\\| g(x, P, h) - \\nabla F(x) \\|}{\\| \\nabla F(x)  \\|}$", fontsize=16)
fig.tight_layout()
if fun_name == 'sparse_sum':
    fig.savefig(f"./{fun_name}_d_{d}_s_{s}.pdf", bbox_inches='tight')
else:
    fig.savefig(f"./{fun_name}_d_{d}.pdf", bbox_inches='tight')
plt.close(fig)