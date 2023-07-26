import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

lw = 3
alpha = 0.36
fontsize = 20 
ax_fontsize = 16
legend_fontsize = 18


class ConvCriterion:
    def __init__(self, tau, f_L):
        self.tau = tau
        self.f_L = f_L
        self.ratios = []
        
    def __call__(self, k, g_k, f_k, f_0):
        ratio = ((f_k - self.f_L) / (f_0 - self.f_L) ).item()
        self.ratios.append(ratio)
        return ratio <= self.tau



def plot_comparison(title, mean_value, std_value, x_label, y_label, labels, out_file, figsize=(10, 7), plot_legend = False, xscale = None, yscale=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=fontsize)
    for i in range(len(labels)):
        ax.plot(range(mean_value[i].shape[0]), mean_value[i], '-', lw=lw, label=labels[i])
        ax.fill_between(range(mean_value[i].shape[0]), mean_value[i] - std_value[i], mean_value[i] + std_value[i], alpha=alpha)
    ax.set_xlabel(x_label, fontsize=ax_fontsize)
    ax.set_ylabel(y_label, fontsize=ax_fontsize)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.legend(fontsize=legend_fontsize)
    fig.savefig("./{}.pdf".format(out_file), bbox_inches='tight')