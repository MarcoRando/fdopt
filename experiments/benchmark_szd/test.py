import torch
import os
from benchmark_functions import SphereFunction

from eszd import 

os.makedirs("./results/impact_of_directions", exist_ok=True)

d = int(1e5)
l = d//2

target = SphereFunction(d)


x = torch.zeros(1, d)
print(target(x).item())