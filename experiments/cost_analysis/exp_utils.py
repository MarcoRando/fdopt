"""Full-matrix (explicit) constructions of the direction families, used ONLY by the experiments.

`ExplicitOperator` has the same interface as the library operators (resample / perturb / accumulate / finalize /
norm_const / d / l), so `fd.FFD` runs on it unchanged: the whole l x d matrix is built in `resample()` and sliced.
`generate()` builds one matrix (what Experiment 1 times).  Families: the library ones plus Haar/QR and QR variants.
"""
from math import floor, log2, pi

import torch

SEED = 121212
IID = {"gaussian", "rademacher"}
ORTHOGONAL = {"coordinate", "qr", "householder", "permuted_householder", "butterfly"}
VARIANTS = {
    "qr_reorth": ("qr", {"reorth": 1}),
    "qr_f32_reorth": ("qr", {"reorth": 1, "gen_dtype": torch.float32}),
}
BASE_FAMILIES = ["coordinate", "qr", "householder", "permuted_householder", "butterfly",
                 "gaussian", "spherical", "rademacher"]
EXPLICIT_FAMILIES = BASE_FAMILIES + list(VARIANTS)


def _gaussian(n, d, gen, dtype, device):
    return torch.randn(n, d, dtype=dtype, generator=gen, device=device)


def _spherical(n, d, gen, dtype, device):
    P = torch.randn(n, d, dtype=dtype, generator=gen, device=device)
    return P.div_(P.norm(dim=1, keepdim=True))


def _rademacher(n, d, gen, dtype, device):
    B = torch.randint(0, 2, (n, d), dtype=torch.int8, generator=gen, device=device)
    return (2 * B - 1).to(dtype)


def _unit_vector(d, gen, dtype, device):
    v = torch.randn(d, generator=gen, dtype=dtype, device=device)
    return v / torch.linalg.norm(v)


def _haar_q(A):
    H, tau = torch.geqrf(A)
    Q = torch.linalg.householder_product(H, tau)
    return Q * torch.where(H.diagonal() >= 0, 1.0, -1.0).to(Q.dtype)


class ExplicitOperator:
    def __init__(self, name, d, l=1, dtype=torch.float64, device="cpu", seed=SEED, label=None,
                 store_identity=True, reorth=0, gen_dtype=None):
        self.name, self.label = name, label or name
        self.d, self.l, self.dtype, self.device = d, l, dtype, device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.store_identity, self.reorth, self.gen_dtype = store_identity, reorth, gen_dtype
        self.P = None
        self.I = None
        if name == "coordinate" and store_identity:
            self.I = torch.eye(d, dtype=dtype, device=device)
        if name == "householder":
            self.I = torch.eye(d, l, dtype=dtype, device=device)
        self.generate = getattr(self, f"_gen_{name}")

    @property
    def iid(self):
        return self.name in IID

    @property
    def orthogonal(self):
        return self.name in ORTHOGONAL

    @property
    def norm_const(self):
        return 1.0 / self.l if self.iid else self.d / self.l

    def resample(self):
        self.P = None
        self.P = self.generate()

    def perturb(self, x, h, a, e):
        return torch.add(x, self.P[a:e], alpha=h)

    def accumulate(self, g, s, a, e):
        g.view(-1).add_(s @ self.P[a:e])

    def finalize(self, g):
        pass

    def release(self):
        self.P = None

    def materialize(self):
        z = torch.zeros(1, self.d, dtype=self.dtype, device=self.device)
        return self.perturb(z, 1.0, 0, self.l)

    def __call__(self):
        self.resample()
        return self

    def _ar(self, n):
        return torch.arange(n, device=self.device)

    def __str__(self):
        return f"Explicit_{self.label}_{self.d}_{self.l}"

    def _gen_coordinate(self):
        if self.store_identity and self.l == self.d:
            return self.I
        inds = torch.randperm(self.d, generator=self.generator, device=self.device)[: self.l]
        if self.store_identity:
            return self.I[inds, :]
        P = torch.zeros((self.l, self.d), dtype=self.dtype, device=self.device)
        P[self._ar(self.l), inds] = 1.0
        return P

    def _gen_qr(self):
        gd = self.gen_dtype or self.dtype
        Q = _haar_q(torch.randn(self.d, self.l, generator=self.generator, dtype=gd, device=self.device))
        for _ in range(self.reorth):
            Q = _haar_q(Q)
        return Q.T.to(self.dtype)

    def _gen_householder(self):
        v = _unit_vector(self.d, self.generator, self.dtype, self.device)
        return (self.I - 2 * v.outer(v[: self.l])).T

    def _gen_permuted_householder(self):
        inds = torch.randperm(self.d, generator=self.generator, device=self.device)[: self.l]
        v = _unit_vector(self.d, self.generator, self.dtype, self.device)
        I = torch.zeros(self.d, self.l, device=self.device, dtype=self.dtype)
        I[inds, self._ar(self.l)] = 1.0
        return (I - 2 * v.outer(v[inds])).T

    def _gen_butterfly(self):
        B = torch.ones(1, 1, dtype=self.dtype, device=self.device)
        num_iters = floor(log2(self.d))
        thetas = 2 * pi * torch.rand(num_iters, generator=self.generator, device=self.device, dtype=self.dtype)
        for i in range(num_iters):
            c, s = torch.cos(thetas[i]) * B[:, : self.l], torch.sin(thetas[i]) * B[:, : self.l]
            B = torch.hstack((torch.vstack([c, s]), torch.vstack([-s, c])))
        B = B[:, : self.l]
        if self.d == B.shape[0]:
            return B[:, : self.l].T
        d_eff, d_0 = B.shape[1], B.shape[0]
        to_add = self.d - B.shape[0]
        B = torch.vstack([B, torch.zeros(to_add, B.shape[1], dtype=self.dtype, device=self.device)])
        B = torch.hstack([B, torch.zeros(B.shape[0], to_add, dtype=self.dtype, device=self.device)])
        B[range(d_0, d_0 + to_add), range(d_eff, d_eff + to_add)] = 1.0
        B = B[:, torch.randperm(B.shape[1], generator=self.generator, device=self.device)[: self.l]]
        return B.T

    def _gen_gaussian(self):
        return _gaussian(self.l, self.d, self.generator, self.dtype, self.device)

    def _gen_spherical(self):
        return _spherical(self.l, self.d, self.generator, self.dtype, self.device)

    def _gen_rademacher(self):
        return _rademacher(self.l, self.d, self.generator, self.dtype, self.device)


def make_explicit(name, d, l, dtype, device, seed=SEED, **kwargs):
    if name in VARIANTS:
        base, extra = VARIANTS[name]
        return ExplicitOperator(base, d, l, dtype, device, seed, label=name, **{**extra, **kwargs})
    return ExplicitOperator(name, d, l, dtype, device, seed, **kwargs)