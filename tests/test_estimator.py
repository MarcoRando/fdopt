import torch

import time
def target(x):
    return 0.5 * torch.linalg.norm(x, dim=1, keepdim=True, ord=2).square()#.sum(dim=1, keepdims=True)

def grad(x):
    return (x)#torch.linalg.norm(x, dim=1, keepdim=True)


d = 5
l = 5
h = 1e-10
P = torch.randn(d, d, dtype=torch.float64)
#P /= torch.linalg.norm(P, dim=0, ord=2)
#P = P.T

i_time = time.time()
A = P.triu()
A = A - A.T
B = torch.matrix_exp(A)
i_time = time.time() - i_time
print(i_time)


#H2
#h_time = time.time()
#P = torch.linalg.qr(P)[0].T
#h_time = time.time() - h_time
#print("H2 ", h_time)


#print(H)


##v = torch.randn(d, dtype=torch.float64)
##v /= torch.linalg.norm(v, ord=2)
###P = (torch.eye(d, d, dtype=torch.float64) - 2 * torch.outer(v, v))[:, :l].T
##print(torch.linalg.norm(P[0]))
##print(P)
#
#
#
#x = torch.ones((1, d), dtype=torch.float64)# * 200
#gamma = 0.6
#
#vals_2 = torch.zeros((1, d), dtype=torch.float64)
#loop_time = time.time()
#fx = target(x)
#for i in range(l):
#    vals_2 += ((target(x.add(P[i], alpha=h)) - fx)/h) * P[i]
#    
#vals_2 *= (d/l)
#loop_time = time.time() - loop_time
#print("LOOP", loop_time)
## EFF
#xn = x.clone()
#add_time = time.time()
#vals = torch.zeros((l, d), dtype=torch.float64)
#
#vals = vals.addcmul_((target(torch.add(x, P, alpha=h))- target(x)), P).div_(h).sum(dim=0).mul(d/l)
#
##xn.addcmul_(value=-gamma)
##vals = (target(torch.add(x, P, alpha=h))- target(x)).mul(P)
##vals.div_(h)
#add_time = time.time() - add_time
#print("ADDCMUL", add_time)
#
#v2 = (target(torch.add(x, P, alpha=h))- target(x))
#v3 = torch.zeros((l, d), dtype=torch.float64)
#for i in range(v2.shape[0]):
##    print(v2[i], P[i])
#    v3[i] =  v2[i] * P[i]
#    
#v3 /=h
#v3 *=d/l
#
#print((v3.sum(dim=0)).allclose(vals))
#print((vals_2).allclose(vals))
#if not (vals_2).allclose(vals):
#    print(abs(vals_2 - vals).max())
#
#
## EFF
#add_time = time.time()
##vals = torch.zeros((3, d))
#
##vals.addcmul_((target(torch.add(x, P, alpha=h))- target(x)), P)
#vals = (target(x.add(P, alpha=h))- target(x)).mul(P).div(h).sum(dim=0, keepdims=True).mul(d/l)#.div(h)
##vals.div_(h)
#add_time = time.time() - add_time
#print("ADDMUL", add_time)
#
#print((v3.sum(dim=0)).allclose(vals))
#print((vals_2).allclose(vals))
#if not (vals_2).allclose(vals):
#    print(abs(vals_2 - vals).max())
##print(vals_2-vals)
#
##print((vals - vals_2 == 0.0).all())
##print(vals- vals_2)
#
##print(vals)
##print(vals_2)
#print(torch.linalg.norm(vals - grad(x))**2)
#print(torch.linalg.norm(vals_2 - grad(x))**2)
#
#
#