"""Direction operators for finite-difference gradient estimation (implicit application only).

A directions object represents P in R^{l x d} (rows = directions) through its ACTION; the matrix is never built
(chunks of at most b rows are, for families without a cheaper closed form).

Interface used by `fd.FFD`:
    resample()                  draw a new set of l directions (O(d) state)
    perturb(x, h, a, e)         (e-a, d) points x + h * P[a:e]
    accumulate(g, s, a, e)      g += s @ P[a:e]   (g is (d,) or (1, d), contiguous, updated in place)
    finalize(g)                 call once after the last accumulate of an estimator (deferred rank-one / butterfly work)
    release()                   drop the sampled state
"""
from math import floor, log2, pi

import torch

SEED = 121212


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


class Operator:
    iid = False
    orthogonal = False

    def __init__(self, d, l=1, dtype=torch.float64, device="cpu", seed=SEED):
        self.d, self.l, self.dtype, self.device = d, l, dtype, device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self._cached = None
        self._alpha = None
        self._u = None

    @property
    def norm_const(self):
        return 1.0 / self.l if self.iid else self.d / self.l

    def resample(self):
        raise NotImplementedError

    def perturb(self, x, h, a, e):
        raise NotImplementedError

    def accumulate(self, g, s, a, e):
        raise NotImplementedError

    def finalize(self, g):
        pass

    def release(self):
        self._cached = None
        self._alpha = None
        self._u = None

    def materialize(self):
        z = torch.zeros(1, self.d, dtype=self.dtype, device=self.device)
        return self.perturb(z, 1.0, 0, self.l)

    def __call__(self):
        self.resample()
        return self

    def _ar(self, n):
        return torch.arange(n, device=self.device)

    def __str__(self):
        return f"{type(self).__name__}_{self.d}_{self.l}"


class CoordinateDirections(Operator):
    orthogonal = True

    def resample(self):
        self.inds = torch.randperm(self.d, generator=self.generator, device=self.device)[: self.l]

    def perturb(self, x, h, a, e):
        X = x.expand(e - a, self.d).clone()
        X[self._ar(e - a), self.inds[a:e]] += h
        return X

    def accumulate(self, g, s, a, e):
        g.view(-1).index_add_(0, self.inds[a:e], s)


class Householder(Operator):
    orthogonal = True

    def resample(self):
        self.v = _unit_vector(self.d, self.generator, self.dtype, self.device)
        self._alpha = None

    def perturb(self, x, h, a, e):
        n = e - a
        X = x.expand(n, self.d).clone()
        ar = self._ar(n)
        X[ar, ar + a] += h
        X.addmm_(self.v[a:e].unsqueeze(1), self.v.unsqueeze(0), alpha=-2.0 * h)
        return X

    def accumulate(self, g, s, a, e):
        g.view(-1)[a:e] += s
        w = self.v[a:e] @ s
        self._alpha = w if self._alpha is None else self._alpha + w

    def finalize(self, g):
        if self._alpha is not None:
            g.view(-1).sub_((2.0 * self._alpha) * self.v)
            self._alpha = None


class PermutedHouseholder(Operator):
    orthogonal = True

    def resample(self):
        self.inds = torch.randperm(self.d, generator=self.generator, device=self.device)[: self.l]
        self.v = _unit_vector(self.d, self.generator, self.dtype, self.device)
        self._alpha = None

    def perturb(self, x, h, a, e):
        n = e - a
        idx = self.inds[a:e]
        X = x.expand(n, self.d).clone()
        X[self._ar(n), idx] += h
        X.addmm_(self.v[idx].unsqueeze(1), self.v.unsqueeze(0), alpha=-2.0 * h)
        return X

    def accumulate(self, g, s, a, e):
        idx = self.inds[a:e]
        g.view(-1).index_add_(0, idx, s)
        w = self.v[idx] @ s
        self._alpha = w if self._alpha is None else self._alpha + w

    def finalize(self, g):
        if self._alpha is not None:
            g.view(-1).sub_((2.0 * self._alpha) * self.v)
            self._alpha = None


class ButterflyMatrix(Operator):
    orthogonal = True

    def resample(self):
        self.m = floor(log2(self.d))
        self.d0 = 2 ** self.m
        self.to_add = self.d - self.d0
        th = 2 * pi * torch.rand(self.m, generator=self.generator, device=self.device, dtype=self.dtype)
        self.cos, self.sin = torch.cos(th), torch.sin(th)
        self._u = None
        self.d_eff = min(self.d0, self.l)
        if self.to_add == 0:
            self.sel = torch.arange(self.l, device=self.device)
        else:
            self.sel = torch.randperm(self.d_eff + self.to_add, generator=self.generator, device=self.device)[: self.l]

    def _half(self, js, ks):
        n = js.shape[0]
        col = torch.ones(n, 1, dtype=self.dtype, device=self.device)
        for k in ks:
            jk = (js >> k) & 1
            c, s = self.cos[k], self.sin[k]
            r = torch.stack([torch.where(jk == 0, c, -s), torch.where(jk == 0, s, c)], dim=1)
            col = (r.unsqueeze(2) * col.unsqueeze(1)).reshape(n, -1)
        return col

    def _bfly_cols(self, js):
        mid = self.m // 2
        lo = self._half(js, range(mid))
        hi = self._half(js, range(mid, self.m))
        return (hi.unsqueeze(2) * lo.unsqueeze(1)).reshape(js.shape[0], -1)

    def _apply_B(self, u):
        T = u
        for k in range(self.m):
            T = T.reshape(-1, 2, 2 ** k)
            t0, t1 = T[:, 0], T[:, 1]
            c, s = self.cos[k], self.sin[k]
            T = torch.stack((c * t0 - s * t1, s * t0 + c * t1), dim=1)
        return T.reshape(-1)

    def chunk(self, a, e):
        sel = self.sel[a:e]
        Pc = torch.zeros(e - a, self.d, dtype=self.dtype, device=self.device)
        if self.to_add == 0:
            Pc[:, : self.d0] = self._bfly_cols(sel)
            return Pc
        is_b = sel < self.d_eff
        Pc[:, : self.d0] = self._bfly_cols(sel.clamp(max=self.d_eff - 1)) * is_b.unsqueeze(1).to(self.dtype)
        off = sel - self.d_eff
        Pc[:, self.d0:] = (off.unsqueeze(1) == self._ar(self.to_add).unsqueeze(0)).to(self.dtype)
        return Pc

    def perturb(self, x, h, a, e):
        return torch.add(x, self.chunk(a, e), alpha=h)

    def accumulate(self, g, s, a, e):
        sel = self.sel[a:e]
        if self._u is None:
            self._u = torch.zeros(self.d0, dtype=self.dtype, device=self.device)
        if self.to_add == 0:
            self._u.index_add_(0, sel, s)
        else:
            is_b = sel < self.d_eff
            self._u.index_add_(0, sel.clamp(max=self.d_eff - 1), s * is_b.to(s.dtype))
            g.view(-1)[self.d0:].index_add_(0, (sel - self.d_eff).clamp(min=0), s * (~is_b).to(s.dtype))

    def finalize(self, g):
        if self._u is not None:
            g.view(-1)[: self.d0].add_(self._apply_B(self._u))
            self._u = None


class RandomRows(Operator):
    draw = None

    def __init__(self, d, l=1, dtype=torch.float64, device="cpu", seed=SEED, cache=True):
        super().__init__(d, l, dtype, device, seed)
        self.cache = cache
        self.base = 0

    def resample(self):
        self._cached = None
        self.base = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=self.generator, device=self.device).item())

    def chunk(self, a, e):
        gen = torch.Generator(device=self.device).manual_seed(self.base + a)
        return self.draw(e - a, self.d, gen, self.dtype, self.device)

    def perturb(self, x, h, a, e):
        Pc = self.chunk(a, e)
        if self.cache:
            self._cached = Pc
        return torch.add(x, Pc, alpha=h)

    def accumulate(self, g, s, a, e):
        Pc = self._cached[: e - a] if (self.cache and self._cached is not None) else self.chunk(a, e)
        g.view(-1).add_(s @ Pc)


class GaussianDirections(RandomRows):
    iid = True
    draw = staticmethod(_gaussian)


class SphericalDirections(RandomRows):
    draw = staticmethod(_spherical)


class RademacherDirections(RandomRows):
    iid = True
    draw = staticmethod(_rademacher)


FAMILIES = {
    "coordinate": CoordinateDirections,
    "householder": Householder,
    "permuted_householder": PermutedHouseholder,
    "butterfly": ButterflyMatrix,
    "gaussian": GaussianDirections,
    "spherical": SphericalDirections,
    "rademacher": RademacherDirections,
}


def make(name, d, l, dtype, device, seed=SEED, **kwargs):
    return FAMILIES[name](d=d, l=l, dtype=dtype, device=device, seed=seed, **kwargs)