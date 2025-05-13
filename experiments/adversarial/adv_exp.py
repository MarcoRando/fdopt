

import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from math import floor
import numpy as np 

import gzip
import matplotlib.pyplot as plt

from torch.utils.data import SubsetRandomSampler

import torch.nn.functional as F

from net import MNISTNet
from target_fun import AdversarialTarget


from fdopt.direction_generators import * #Householder, ButterflyMatrix, CoordinateDirections, GaussianDirections, SphericalDirections, QRDirections
from fdopt.optimizers import LSFDOpt
from fdopt.grad_estimators import ForwardFD


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])
batch_size=128
seed = 12131415
device = 'cuda'
dtype = torch.float64
generator = torch.Generator(device=device).manual_seed(seed)

out_dir = sys.argv[1]
reps = int(sys.argv[2])

method = sys.argv[3]


os.makedirs(f"{out_dir}/adv_exp/mnist/fun_values", exist_ok=True)
os.makedirs(f"{out_dir}/adv_exp/mnist/distortion", exist_ok=True)
os.makedirs(f"{out_dir}/adv_exp/mnist/images", exist_ok=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


net = MNISTNet(device=device,dtype=dtype)


net.load_state_dict(torch.load("./models/mnist/student", map_location=device))
net.eval()

d = 784
#l = d//2

methods = {
    'spherical' : SphericalDirections,
    'gaussian' : GaussianDirections,
    'rademacher' : RademacherDirections,
    'coordinate' : CoordinateDirections,
    'householder' : Householder,
    'butterfly' : ButterflyMatrix,
    "permuted_householder" : PermutedHouseholder,
    'qr' : QRDirections,
}



def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).copy()
        data = torch.from_numpy(data).view(num_images, 1, 28, 28)
        data = (data / 255) - 0.5
        return data.to(dtype)

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).copy()
    return torch.from_numpy(np.arange(10) == labels[:, None]).to(dtype)

test_x = extract_data('./data/MNIST/raw/t10k-images-idx3-ubyte.gz', 10000)
test_y = extract_labels('./data/MNIST/raw/t10k-labels-idx1-ubyte.gz', 10000)


# Currently used
# lam = 1.0
# theta = 0.33#9
# T = 30000

lam = 100.0
theta = 0.5
T = 30000


# 
label, matrix_cls = method, methods[method]
for l in [d, round((4/5) * d), d//2, d//3]:# [2, d//3, d//2, round((4/5) * d),  d]:

    results = []#{label : [] for label in methods.keys()}

    values = []# {label : [] for label in methods.keys()}
    gen = torch.Generator(device=device).manual_seed(seed)
    for i in range(0, 100):
        x, y = test_x[i], test_y[i].argmax()
        if net(x.to(device=device, dtype=dtype).view(1, 1, 28, 28)).argmax(1).cpu().item() != y:
            print(f"[--] Skipping {i} since it is not the correct class")
            break
        curr_im = x.clone().to(dtype=dtype, device=device)

        print(f"[{i}] Solving for ", label)
        P = matrix_cls(d = d, l = l, dtype=dtype, device=device, seed=seed)
        g = ForwardFD(norm_const=1/l) if label=='gaussian' or label=='rademacher' else ForwardFD(norm_const=d/l)
        

        target = AdversarialTarget(net, curr_im, y, lam=lam, cw=False)

        opt = LSFDOpt(P=P, g=g, gamma_0=1.0, h=1e-7, gamma_minmax=(1e-10, 1000.0), theta=theta, rho=1/theta, c = 1e-7)
        v =  torch.full((1, 784), 10.0,  dtype=dtype, device=device) #+  torch.rand((1, 784), generator=gen, dtype=dtype, device=device) 
        # while target(v) == torch.inf:
        #     v = torch.randn((1, 784), generator=generator, dtype=dtype, device=device) - 0.5
        #     print("TARGET(v_0) = ",target(v))
        if target(v) == torch.inf:
            break
        avg_success = []
        avg_fvals = []
        avg_dist = []
        avg_norm_dist = []
        init_distortion = target.get_distortion(v).flatten().cpu().item()
        for rep_idx in range(reps):
            v_opt, f_values, num_evals, gammas, it_times = opt.optimize(target, v.clone(), T=T)
            im_pert = target.apply_perturbation(v_opt)
            y_predicted = F.softmax(target.net(im_pert.view(1, 1, 28, 28)), dim=1).argmax(1).cpu().item()
            distortion = target.get_distortion(v_opt).flatten().cpu().item()
            avg_success.append(int(y_predicted != y))
            avg_dist.append(distortion)
            avg_fvals.append(f_values)
            avg_norm_dist.append(distortion / init_distortion)
#            print(f"[{label}] {i},{avg_norm_dist[-1]},{avg_norm_dist[-1]},{avg_fvals[-1][-1]},{y_predicted != y}\n")
            with open(f"{out_dir}/adv_exp/mnist/fun_values/fun_values_mnist_{label}_{l}.log", 'a') as f:
                for j in range(len(f_values)):
                    f.write(f"{i},{f_values[0]},{f_values[j]},{num_evals[j]},{rep_idx},{it_times[j]}\n")
                    f.flush()
        print(f"[--] Average Success = {np.mean(avg_success)}")
        with open(f"{out_dir}/adv_exp/mnist/distortion/fun_values_mnist_{label}_{l}.log", 'a') as f:
            for j in range(reps):
                f.write(f"{i},{avg_dist[j]},{avg_norm_dist[j]},{avg_success[j]}\n")
                f.flush()
#        print(f"y_predicted = {y_predicted}")
#        print(avg_fvals)
#        results.append(np.mean(avg_success))
#        values.append(np.mean(avg_fvals, axis=0))
#        print(f"[{label}] avg_success = {results[-1]}\tavg (last) value = {values[-1][-1]}\t dist = {np.mean(avg_dist)} +/- {np.std(avg_dist)} \tavg normalized distortion = {np.mean(avg_norm_dist)} +/- {np.std(avg_norm_dist)}")


        #v = (v - v.min()) / (v.max() - v.min())

        #print(v)
        if i % 10 ==0:
            fig, ax = plt.subplots(1, 3, figsize=(12,6))
            p_x = F.softmax(target.net(curr_im.view(1, 1, 28, 28)), dim=1)
            ax[0].set_title(f"Original [y = {p_x.argmax(1).cpu().item()}, p = {p_x.max(1).values.cpu().item()}]")
            ax[0].imshow(curr_im.view(28, 28).cpu().detach().numpy(), cmap='gray')
            ax[0].set_axis_off()
            p_v = F.softmax(target.net(im_pert.view(1, 1, 28, 28)), dim=1)
            ax[1].set_title(f"Perturbed [y = {p_v.argmax(1).cpu().item() }, p = {p_v.max(1).values.cpu().item()}]")
            ax[1].imshow(im_pert.view(28, 28).cpu().detach().numpy(), cmap='gray')
            ax[1].set_axis_off()
            ax[2].set_title(f"Perturbation $v$")
            ax[2].imshow(v_opt.view(28, 28).cpu().detach().numpy(), cmap='gray')
            ax[2].set_axis_off()

            fig.tight_layout()
#            fig.savefig(f"./mnist_test_pert_{label}_{l}.png", bbox_inches='tight')
            fig.savefig(f"{out_dir}/adv_exp/mnist/images/mnist_test_pert_{label}_{l}_{i}.png", bbox_inches='tight')
            plt.close(fig)


    # with open(f"{out_dir}/adv_exp/mnist/success_rate_mnist_{l}.log", 'w') as f:
    #     for (label, values) in results.items():
    #         f.write(f"{label},{np.mean(values)},{np.std(values)}\n")
    #         f.flush()
    #         print(f"[--] {label} = {np.mean(values)}")