

import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from math import floor

import matplotlib.pyplot as plt

from torch.utils.data import SubsetRandomSampler

import torch.nn.functional as F

from net import MNISTNet
from target_fun import AdversarialTarget



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])
batch_size=128
seed = 12131415
device = 'cuda'
dtype = torch.float64
generator = torch.Generator(device=device).manual_seed(seed)



test_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)


net = MNISTNet(device=device,dtype=dtype)


net.load_state_dict(torch.load("./models/mnist/student", map_location=device))
net.eval()





x, y = test_set[20]
x = x.to(dtype=dtype, device=device)

target = AdversarialTarget(net, x, y)

v =  -0.65 * torch.randn((2, 784), generator=generator, dtype=torch.float64, device=device) 
im_pert = target.apply_perturbation(v)


ris = target(v)
print("Result: ", ris)
for i in range(2):
    fig, ax = plt.subplots(1, 3, figsize=(12,6))
    ax[0].set_title(f"Original [y = {y}]")
    ax[0].imshow(x.view(28, 28).cpu().detach().numpy(), cmap='gray')
    ax[1].set_title(f"Perturbed [y = {target.net(im_pert[i].view(1, 1, 28, 28)).argmax(1).cpu().item() }]")
    ax[1].imshow(im_pert[i].view(28, 28).cpu().detach().numpy(), cmap='gray')
    ax[2].set_title(f"Perturbation $v$")
    ax[2].imshow(v[i].view(28, 28).cpu().detach().numpy(), cmap='gray')
    fig.tight_layout()
    fig.savefig(f"./mnist_test_pert_{i}.png", bbox_inches='tight')
    plt.close(fig)
