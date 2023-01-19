import time
import torch
#import torch_butterfly as tbtt
import torch_dct as dct

n = int(2**14)
print("d: {}".format(n))
normalized = True
#b = tbtt.special.dct(n, type=2, normalized=normalized)
v = torch.eye(n, n, device = 'cuda')
#b(v)
#btt_time = time.time()
#P_butt = b(v)
#btt_time = time.time() - btt_time
dct.dct(v, norm='ortho')
dct_time = time.time()
P_dct = dct.dct(v, norm='ortho')
dct_time = time.time() - dct_time

gtime = time.time()
P_g = torch.randn(n, n, device = 'cuda')
gtime = time.time() - gtime

#print("[--] Butterfly DCT: {}".format(btt_time))
print("[--] DCT: {}".format(dct_time))
print("[--] Gaussian: {}".format(gtime))
