import numpy as np
import torch
from sszd.direction_matrices import DirectionMatrix
from concurrent.futures import ProcessPoolExecutor
from numba import jit

class NumbaDirections:
    
    def __init__(self, d, l, seed, dtype = np.float64) -> None:
        self.d = d
        self.l = l
    
    def _compute_truncated_Q(self, A):
        Q = torch.zeros((self.d, self.l), dtype=self.dtype, device=self.device)
        for k in range(self.l):
            pr_sum = torch.zeros(self.d, dtype=self.dtype, device=self.device)
            for i in range(k):
                pr_sum +=A[:, k].dot(Q[:, i])/torch.linalg.norm(Q[:, i]) * Q[:, i]
            Q[:, k] = A[:, k] - pr_sum
            Q[:,k] = Q[:,k] / torch.linalg.norm(Q[:,k])
        return -Q
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        return self.nrm_const * self._compute_truncated_Q(A)
        


class GSDirections(DirectionMatrix):
    
    #def compute_sum(A_vec, Q_vec):
    
    def _compute_truncated_Q(self, A):
        Q = torch.zeros((self.d, self.l), dtype=self.dtype, device=self.device)
        for k in range(self.l):
            pr_sum = torch.zeros(self.d, dtype=self.dtype, device=self.device)
            for i in range(k):
                pr_sum +=A[:, k].dot(Q[:, i])/torch.linalg.norm(Q[:, i]) * Q[:, i]
            Q[:, k] = A[:, k] - pr_sum
            Q[:,k] = Q[:,k] / torch.linalg.norm(Q[:,k])
        return -Q
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        return self.nrm_const * self._compute_truncated_Q(A)
        
class SRDirections(DirectionMatrix):
    def _compute_truncated_Q(self, A):
        Q = A.clone()
        for k in range(self.l):
            for i in range(k):
                Q[:, k] = Q[:, k] - Q[:, i].dot(Q[:, k]) * Q[:, i]
            Q[:,k] = Q[:,k] / torch.linalg.norm(Q[:,k])
        return -Q
    
    def __call__(self):
        A = torch.randn(size=(self.d, self.l), generator=self.generator, device=self.device, dtype=self.dtype)
        return self.nrm_const * self._compute_truncated_Q(A)
