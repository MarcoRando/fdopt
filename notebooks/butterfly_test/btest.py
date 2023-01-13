import torch

from torch_butterfly.butterfly import Butterfly

nblocks = 4
device='cpu'
in_size, out_size = 2, int(1e4)
b = Butterfly(in_size, out_size, True, 
              complex=False, increasing_stride=False,
              init='identity', nblocks=nblocks).to(device)

in_rnd = torch.zeros(6, 2)#randn(2, in_size)

P = b(in_rnd)
#Pt = b.forward(in_rnd, transpose=True).t()

print(P)