import numpy as np
import torch
from math import cos, sin, pi, log2, sqrt
import matplotlib.pyplot as plt
from matplotlib import cm

from eszd.directions import RandomCoords, DirectionMatrix

def plot_comp_time(results, dims, title = "Time comparison", xlab="$d$", xscale = "log", out="./"):
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(title, fontsize=20)
    c = 1
    for (label, res) in results.items():    
        color =   cmap((c - 0.01)/len(results))
        mu = [x[0] for x in res]
        std = [x[1] for x in res]
        ax.plot(dims[:len(mu)], mu, '-', label=label, c=color, lw=3)
        ax.errorbar(dims[:len(mu)], mu, std, marker='o', ms=10,color=color, mew=4)
        c+=1
    ax.legend(loc='best', fontsize=16)
    ax.set_xlabel(xlab, fontsize=18)
    ax.set_ylabel("$s$", fontsize=18)
    if xscale is not None:
        ax.set_xscale(xscale)
    ax.set_yscale("log")
    fig.savefig("{}.pdf".format(out), bbox_inches='tight')
    plt.close(fig)
    
    

class RotCoords(RandomCoords):
    def __init__(self, l, use_identity=False, mul_seed=12, change_every=0, mul_device='cpu', **kwargs):
        super().__init__(mul_seed, mul_device, l=l, **kwargs)
        self.A = None
        self.g_inds = torch.ones(l, device=mul_device)
        self.use_identity = use_identity
        self.change_every = change_every
        self.t = 0
        if use_identity:
            self.A = torch.eye(self.d, self.l, device=self.device, dtype=self.dtype)
        else:
            self._build_random_coords()

    def _build_random_coords(self):
        self.A = torch.zeros((self.d, self.l), device=self.device, dtype=self.dtype)
        inds = self.inds.multinomial(self.l, replacement=False, generator=self.mgen)
        self.A[inds, range(self.l)] = ((torch.rand(self.l, device=self.device, generator=self.generator) < 0.5)*2 - 1).to(dtype=self.dtype) 

    def __call__(self):
        theta = 2 * pi * torch.rand(1, dtype=self.dtype,generator=self.mgen)
        c, s = torch.cos(theta), torch.sin(theta)
        if not self.use_identity and self.change_every > 0 and self.t > 0 and self.t % self.change_every == 0:
            self._build_random_coords()
        # TODO: Handle l = 1 case            
        i, j =  self.g_inds.multinomial(2, replacement=False, generator=self.mgen)
        #print(i, j, self.g_inds.shape)
        i, j = (i, j) if i < j else (j, i)  
        self.A[i, i], self.A[i, j] = c, -s    
        self.A[j, i], self.A[j, j] = s, c
        #self.I = A.mm(self.I)
        return self.nrm_const * self.A


class ReflectedDirections(DirectionMatrix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.I = torch.eye(self.d, self.l, dtype=self.dtype, device=self.device)


    def __call__(self):
        x = torch.rand(self.d, generator=self.generator, dtype=self.dtype, device=self.device)
        u = x + torch.dot(x, x)
        v = u / u.norm()
        return self.nrm_const * (self.I - 2.0 * torch.outer(v, v)[:self.l, :].T)
    
class ButterflyDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
    def __call__(self):
        thetas = 2 * pi * torch.rand(int(log2(int(self.d))), generator=self.generator, device=self.device)
        A = torch.eye(1, dtype=self.dtype, device=self.device)
        k = int(log2(self.d))
        num_additional = self.d - 2**k
        for i in range(k):
            theta = thetas[i]
            A_cut = A[:, :self.l]
            c, s = cos(theta) * A_cut, sin(theta) * A_cut
            A = torch.vstack((torch.hstack([c, -s]), torch.hstack([s, c]))) #torch.hstack([c, -s, s, c]).reshape(2*c.shape[0], -1)

        A = A[:, :self.l]
        if num_additional == 0:
            return self.nrm_const * A
        A = torch.vstack((A, torch.zeros(num_additional, A.shape[1], dtype=self.dtype, device=self.device)))
        A = torch.hstack((A, torch.zeros(A.shape[0], num_additional, dtype=self.dtype, device=self.device)))
        idx =range(-1, -num_additional -1, -1)
        A[idx, idx] = 2 * (torch.rand(num_additional, dtype=self.dtype, generator=self.generator, device=self.device) > 0.5).to(self.dtype) -1.0
        #for i in range(1, num_additional + 1):
        #    A[-i, -i] = 2 * (torch.rand(1, dtype=self.dtype, generator=self.generator) > 0.5).to(self.dtype) -1.0
        inds = torch.randperm(A.shape[1], device=self.device, generator=self.generator)
        A = A[:, inds]
        return self.nrm_const * A[:, :self.l]
#        return self._bprod(v)

            
class CirculantDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def __call__(self):
        tensor =torch.rand(self.d, dtype=self.dtype, device=self.device, generator=self.generator)
        tmp = torch.cat([tensor.flip((0,)), torch.narrow(tensor.flip((0,)), dim=0, start=0, length=tensor.shape[0]-1)], dim=0)
        return tmp.unfold(0, tensor.shape[0], 1).flip((-1,))
            
class PermutedDirections(DirectionMatrix):
    def __init__(self, p_seed: int = 1212, **kwargs):
        super().__init__(**kwargs)
        self.pgen = torch.Generator()
        self.pgen.manual_seed(p_seed)
        self.prm = torch.randperm(self.d, generator=self.pgen)        
        self.lb, self.ub = 0, self.l 
        self.I = torch.eye(self.d, device=self.device)
        
    def __call__(self):
        if self.lb < self.d and self.ub <= self.d:
            P = self.nrm_const * self.I[:, self.prm[self.lb: self.ub]]
            self.lb = self.ub
            self.ub += self.l
        elif self.lb < self.d and self.ub > self.d:
            diff = self.ub - self.d
            P = self.nrm_const * torch.vstack((self.I[:, self.prm[self.lb: self.d]], self.I[:, self.prm[0: diff]])) 
            self.lb = diff
            self.ub = diff + self.l
        elif self.lb == self.d:
            P = self.nrm_const * self.I[:, self.prm[0: self.l]]
            self.lb = self.l
            self.ub = self.lb + self.l
        return P    
