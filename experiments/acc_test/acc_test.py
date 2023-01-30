import torch
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x.square().sum(keepdims=True, dim=1)

def grad(x):
    return 2 * x.sum(keepdims=True, dim=1)


X = torch.linspace(-1, 1, 100).reshape(-1, 1)
Y = f(X)

T = 100
d = 2
x = torch.tensor([[500.0 for _ in range(d)]])
x_gd = x.clone()
y = x
evals = [f(x).item()]
evals_gd = [f(x).item()]

alpha = 0.01
beta = 1



for t in range(T):
    x_n = y - alpha * grad(x)
    beta_n = (1 + np.sqrt(1 + 4 * np.square(beta)))/2
    y = x_n + ((beta - 1)/beta_n) * (x_n - x)
    x = x_n
    beta = beta_n
    x_gd = x_gd - alpha * grad(x_gd)
    evals.append(f(y).item())
    evals_gd.append(f(x_gd).item())
    
    

fig, ax =plt.subplots()

ax.plot(range(len(evals)), evals, '-')
ax.plot(range(len(evals_gd)), evals_gd, '-')
ax.set_yscale("log")
fig.savefig("./test_acc.pdf")
plt.close(fig)