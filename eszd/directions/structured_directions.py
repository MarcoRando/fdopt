import os
import torch
from eszd.directions.directions import DirectionMatrix
from torch.utils.cpp_extension import load

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
        self.I = torch.eye(self.d, self.l, device=self.device, dtype=self.dtype)

    def __call__(self):
        v = torch.randn(self.d, generator=self.generator, dtype=self.dtype)
        v /= v.norm()
        return self.nrm_const * (self.I - 2 * torch.outer(v, v[:self.l]))#householder_ref.get_reflector(self.d, self.l)
        

class HouseholderQRDirections(DirectionMatrix):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        return self.nrm_const * torch.linalg.qr(A, mode='reduced')[0]
    
class RandomCoords(DirectionMatrix):
    def __init__(self, mul_seed=12, mul_device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.mgen = torch.Generator(device=mul_device)
        self.mgen.manual_seed(mul_seed)
        self.inds = torch.ones(self.d, device=mul_device)
        self.I = torch.zeros((self.d, self.l), device=self.device, dtype=self.dtype)
    
    def __call__(self):
        if self.d == self.l:
            return self.I
        inds = self.inds.multinomial(self.l, replacement=False, generator=self.mgen)
        return self.nrm_const * self.I[:, inds]