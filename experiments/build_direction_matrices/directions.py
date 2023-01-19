import torch
from math import log2, cos, sin
from eszd.directions import DirectionMatrix


class CirculantDirections(DirectionMatrix):
    
    def __call__(self):
        dim = 0
        tensor = torch.randn(self.d, dtype=self.dtype, device=self.device, generator=self.generator)
        tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=self.l-1)], dim=dim)
        return tmp.unfold(dim, self.l, 1).flip((-1,))
    
    
class SingleHouseholder(DirectionMatrix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.I = torch.eye(self.d, self.l, dtype=self.dtype, device=self.device)

    def __call__(self):
        x = torch.randn(self.d, generator=self.generator, dtype=self.dtype, device=self.device)
        u = x + torch.dot(x, x)
        v = u / u.norm()
        return self.nrm_const * (self.I - 2.0 * torch.outer(v, v)[:self.l, :].T)
    
    
class ButterflyDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
    def __call__(self):
        thetas = 2 * torch.pi * torch.rand(int(log2(int(self.d))), generator=self.generator, device=self.device)
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