

import sys
import torch
import torchvision
import tqdm
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64
generator = torch.Generator(device=device).manual_seed(seed)

training_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# train_size = len(training_set)
# valid_size = 0.2
# split_point = int(floor(valid_size * train_size))
# indices = list(range(train_size))

# tr_idx, vl_idx = indices[split_point:], indices[:split_point]

# tr_sampler = SubsetRandomSampler(tr_idx, generator = torch.Generator().manual_seed(seed))
# vl_sampler = SubsetRandomSampler(vl_idx, generator = torch.Generator().manual_seed(seed))

# training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, sampler=tr_sampler)
# validation_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, sampler=vl_sampler)



net = MNISTNet(device=device,dtype=dtype)


net.load_state_dict(torch.load("./models/mnist/student", map_location=device))
net.eval()




def compute_accuracy(loader):
    net.eval()
    running_loss = 0.0
    iterator = tqdm.tqdm(enumerate(loader))
    for (i, data) in iterator:
        x, y = data
        x = x.to(device).to(torch.float64)
        preds = net(x)
        running_loss += (y == F.softmax(preds.cpu(), 1).max(1)[1]).to(torch.int64).sum().item()
        iterator.set_postfix({'succesfully classified' : running_loss})
    return running_loss
print("Accuracy on Training:", compute_accuracy(training_loader) / len(training_set))    
#print("Accuracy on Validation:", compute_accuracy(validation_loader) / len(vl_idx))    
print("Accuracy on Test:", compute_accuracy(test_loader) / len(test_set))    


