import torch
import matplotlib.pyplot as plt
from targets.synthetic import SphereFunction, LeastSquares


x_star = torch.tensor([3.0],dtype=torch.float64)
target = LeastSquares(m = 1, d = 1, x_star = x_star, dtype=torch.float64)
X = torch.linspace(-1, 5 ,1000).reshape(-1, 1).to(dtype=torch.float64)

y = [target(x).item() for x in X]

fig, ax = plt.subplots()
ax.plot(X, y,'-')
fig.savefig("./test.pdf", bbox_inches='tight')