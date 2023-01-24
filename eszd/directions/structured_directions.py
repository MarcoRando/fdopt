import os
import torch
from eszd.directions.directions import DirectionMatrix

from torch.fft import fft

from torch_dct import dct

from math import pi, sqrt, cos, sin, log2

import numpy as np

VERBOSE = False
def _resolve(name):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), name)

#try:
#    import householder_ref
#except ImportError:
#    if VERBOSE:
#        print('Falling back to JIT compiling householder_ref')
#    torch_householder_cpp = load(
#        name='householder_ref',
#        sources=[
#            _resolve('householder.cpp'),
#        ],
#        verbose=VERBOSE,
#    )

class HouseholderReflection(DirectionMatrix):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
#        self.mgen = torch.Generator(device=self.device)
#        self.mgen.manual_seed(mul_seed)
        self.I = torch.eye(self.d, self.d, device=self.device, dtype=self.dtype)

    def __call__(self):
        v = torch.randn(self.d, generator=self.generator, dtype=self.dtype, device=self.device)
        v /= v.norm()
        inds = torch.randperm(self.d, device=self.device, generator=self.generator)[:self.l]

        return self.nrm_const * (self.I[:,inds] - 2 * torch.outer(v, v[:self.l]))#householder_ref.get_reflector(self.d, self.l)
        

class HouseholderQRDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        return self.nrm_const * torch.linalg.qr(A, mode='reduced')[0]
    
class RandomCoords(DirectionMatrix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.I = torch.eye(self.d, self.d, device=self.device, dtype=self.dtype)
    
    def __call__(self):
        inds = torch.randperm(self.d, device=self.device, generator=self.generator)[:self.l]
        return self.nrm_const * self.I[:, inds]
    
class DCTDirections(DirectionMatrix):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.I = torch.eye(self.d, self.d, device=self.device, dtype=self.dtype)
    
    def __call__(self):
        inds = torch.randperm(self.d, device=self.device, generator=self.generator)[:self.l]

        x = self.nrm_const * self.I[inds, :]
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.view_as_real(fft(v, dim=1))

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)
        return V.T
#        return dct(super().__call__().T, norm='ortho').T
        
        
class ButterflyDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
    def __call__(self):
        thetas = 2 * torch.pi * torch.rand(int(log2(self.d)), generator=self.generator, device=self.device)
        A = torch.ones((1, 1), dtype=self.dtype, device=self.device)
        k = int(log2(self.d))
        num_additional = self.d - 2**k
        for i in range(k):
            theta = thetas[i]
            A_cut = A[:, :self.l]
            c, s = torch.cos(theta) * A_cut, torch.sin(theta) * A_cut
            A = torch.vstack((torch.hstack([c, -s]), torch.hstack([s, c]))) #torch.hstack([c, -s, s, c]).reshape(2*c.shape[0], -1)

#        A = A[:, :self.l]
#        if num_additional == 0:
#            return self.nrm_const * A
#        additional_rows = torch.zeros(num_additional, A.shape[1], dtype=self.dtype, device=self.device)
#        A = torch.vstack((A, additional_rows))
#        A = torch.hstack((A, additional_rows))
#        idx =range(-num_additional, -1, 1)
        
#        A[idx, idx] = 1.0 #2 * (torch.rand(num_additional, dtype=self.dtype, generator=self.generator, device=self.device) > 0.5).to(self.dtype) -1.0
        #for i in range(1, num_additional + 1):
        #    A[-i, -i] = 2 * (torch.rand(1, dtype=self.dtype, generator=self.generator) > 0.5).to(self.dtype) -1.0
#        inds = torch.randperm(A.shape[1], device=self.device, generator=self.generator)[:self.l]
#        A = A[:, inds]
        return self.nrm_const * A