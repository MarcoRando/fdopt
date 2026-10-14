"""Direction operators for finite-difference gradient estimation.

Every family is a single class with an `implicit` option:

* implicit=False : the full (l x d) matrix P is materialised by `resample()` (`generate()`), rows are sliced.
* implicit=True  : only a light state (indices, a vector, angles, a seed) is stored; the rows of a chunk
                   [a, e) are produced on the fly in `perturb()` and applied in `accumulate()`.

Common interface used by `fd.FFD`:
    resample()                  draw a new set of l directions
    perturb(x, h, a, e)         (e-a, d) points x + h * P[a:e]
    accumulate(g, s, a, e)      g += s @ P[a:e]   (s has length e-a; g is (d,) or (1, d), updated in place)
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
    supports_implicit = True

    def __init__(self, d, l=1, dtype=torch.float64, device="cpu", seed=SEED, implicit=False):
        if implicit and not self.supports_implicit:
            raise ValueError(f"{type(self).__name__} has no implicit form")
        self.d, self.l, self.dtype, self.device = d, l, dtype, device
        self.implicit = implicit
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.P = None          # explicit mode: current (l, d) matrix
        self._cached = None    # implicit mode: last chunk (used by `accumulate` to avoid regenerating it)
        self._alpha = None     # implicit Householder variants: pending rank-one coefficient (see finalize)
        self._u = None         # implicit butterfly: pending coefficient vector (see finalize)

    @property
    def norm_const(self):
        return 1.0 / self.l if self.iid else self.d / self.l

    # ---- public interface -------------------------------------------------
    def resample(self):
        if self.implicit:
            self._cached = None
            self._resample_implicit()
        else:
            self.P = None  # free the previous matrix before allocating the next one
            self.P = self.generate()

    def perturb(self, x, h, a, e):
        if self.implicit:
            return self._perturb_implicit(x, h, a, e)
        if self.P is None:
            raise RuntimeError("call resample() first")
        return torch.add(x, self.P[a:e], alpha=h)

    def accumulate(self, g, s, a, e):
        if self.implicit:
            self._accumulate_implicit(g, s, a, e)
        else:
            g.view(-1).add_(s @ self.P[a:e])

    def finalize(self, g):
        """Called once after the last `accumulate` of an estimator (FFD does this); lets an operator defer work
        that is cheaper on the whole vector than chunk by chunk. No-op by default."""

    def release(self):
        """Drop the sampled directions (not the constructor-time state), e.g. before a memory measurement."""
        self.P = None
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
        return f"{type(self).__name__}_{self.d}_{self.l}_{'implicit' if self.implicit else 'explicit'}"

    # ---- to be provided by families ---------------------------------------
    def generate(self):
        raise NotImplementedError("no explicit form")

    def _resample_implicit(self):
        raise NotImplementedError

    def _perturb_implicit(self, x, h, a, e):
        raise NotImplementedError

    def _accumulate_implicit(self, g, s, a, e):
        raise NotImplementedError


class CoordinateDirections(Operator):
    """Random subset of coordinate axes.

    explicit: rows of a stored identity matrix (the "naive" matrix implementation; with l == d the stored identity
    is returned as is). implicit: only the sampled indices are stored; perturbation is an indexed update and the
    accumulation is an index_add_ (no multiplication by a sampling matrix) -- the efficient implementation.
    """
    orthogonal = True

    def __init__(self, d, l=1, dtype=torch.float64, device="cpu", seed=SEED, implicit=False, store_identity=True):
        super().__init__(d, l, dtype, device, seed, implicit)
        self.store_identity = store_identity and not implicit
        self.I = torch.eye(d, dtype=dtype, device=device) if self.store_identity else None

    def _inds(self):
        return torch.randperm(self.d, generator=self.generator, device=self.device)[: self.l]

    def generate(self):
        if self.store_identity and self.l == self.d:
            return self.I
        inds = self._inds()
        if self.store_identity:
            return self.I[inds, :]
        P = torch.zeros((self.l, self.d), dtype=self.dtype, device=self.device)
        P[self._ar(self.l), inds] = 1.0
        return P

    def _resample_implicit(self):
        self.inds = self._inds()

    def _perturb_implicit(self, x, h, a, e):
        X = x.expand(e - a, self.d).clone()
        X[self._ar(e - a), self.inds[a:e]] += h
        return X

    def _accumulate_implicit(self, g, s, a, e):
        g.view(-1).index_add_(0, self.inds[a:e], s)


class QRDirections(Operator):
    """Haar-distributed orthonormal rows via QR of a Gaussian matrix (explicit only).

    reorth    : number of extra QR passes applied to the computed Q (re-orthogonalisation).
    gen_dtype : dtype used for sampling + QR (e.g. float32); the result is cast to `dtype`.
    """
    orthogonal = True
    supports_implicit = False

    def __init__(self, d, l=1, dtype=torch.float64, device="cpu", seed=SEED, implicit=False, reorth=0, gen_dtype=None):
        super().__init__(d, l, dtype, device, seed, implicit)
        self.reorth, self.gen_dtype = reorth, gen_dtype

    def generate(self):
        gd = self.gen_dtype or self.dtype
        P = torch.randn(self.d, self.l, generator=self.generator, dtype=gd, device=self.device)
        Q = self._haar_q(P)
        for _ in range(self.reorth):
            Q = self._haar_q(Q)
        return Q.T.to(self.dtype)

    @staticmethod
    def _haar_q(A):
        """Q factor of A with the sign correction Q <- Q diag(sign(diag R)), i.e. the QR with non-negative
        diagonal of R. Applied to a Gaussian matrix this gives Haar-distributed orthonormal columns (the raw
        Householder QR forces R_ii to a fixed sign convention and is NOT Haar). Applied to an already orthonormal
        matrix (re-orthogonalisation pass) it returns the same matrix up to rounding."""
        H, tau = torch.geqrf(A)
        Q = torch.linalg.householder_product(H, tau)
        diag = H.diagonal()
        return Q * torch.where(diag >= 0, 1.0, -1.0).to(Q.dtype)


class Householder(Operator):
    """First l rows of the reflection I - 2 v v^T (v uniform on the sphere)."""
    orthogonal = True

    def __init__(self, d, l=1, dtype=torch.float64, device="cpu", seed=SEED, implicit=False):
        super().__init__(d, l, dtype, device, seed, implicit)
        self.I = None if implicit else torch.eye(d, l, device=device, dtype=dtype)

    def generate(self):
        v = _unit_vector(self.d, self.generator, self.dtype, self.device)
        return (self.I - 2 * v.outer(v[: self.l])).T

    def _resample_implicit(self):
        self.v = _unit_vector(self.d, self.generator, self.dtype, self.device)
        self._alpha = None  # running sum of v[a:e] . s; the rank-one correction is applied once in finalize

    def _perturb_implicit(self, x, h, a, e):
        n = e - a
        X = x.expand(n, self.d).clone()
        ar = self._ar(n)
        X[ar, ar + a] += h
        X.addmm_(self.v[a:e].unsqueeze(1), self.v.unsqueeze(0), alpha=-2.0 * h)
        return X

    def _accumulate_implicit(self, g, s, a, e):
        g.view(-1)[a:e] += s
        w = self.v[a:e] @ s
        self._alpha = w if self._alpha is None else self._alpha + w

    def finalize(self, g):
        if self.implicit and self._alpha is not None:
            g.view(-1).sub_((2.0 * self._alpha) * self.v)
            self._alpha = None


class PermutedHouseholder(Operator):
    """l random rows (random column subset) of the reflection I - 2 v v^T."""
    orthogonal = True

    def _draw(self):
        inds = torch.randperm(self.d, generator=self.generator, device=self.device)[: self.l]
        v = _unit_vector(self.d, self.generator, self.dtype, self.device)
        return inds, v

    def generate(self):
        inds, v = self._draw()
        I = torch.zeros(self.d, self.l, device=self.device, dtype=self.dtype)
        I[inds, self._ar(self.l)] = 1.0
        return (I - 2 * v.outer(v[inds])).T

    def _resample_implicit(self):
        self.inds, self.v = self._draw()
        self._alpha = None  # running sum of v[idx] . s; the rank-one correction is applied once in finalize

    def _perturb_implicit(self, x, h, a, e):
        n = e - a
        idx = self.inds[a:e]
        X = x.expand(n, self.d).clone()
        X[self._ar(n), idx] += h
        X.addmm_(self.v[idx].unsqueeze(1), self.v.unsqueeze(0), alpha=-2.0 * h)
        return X

    def _accumulate_implicit(self, g, s, a, e):
        idx = self.inds[a:e]
        g.view(-1).index_add_(0, idx, s)
        w = self.v[idx] @ s
        self._alpha = w if self._alpha is None else self._alpha + w

    def finalize(self, g):
        if self.implicit and self._alpha is not None:
            g.view(-1).sub_((2.0 * self._alpha) * self.v)
            self._alpha = None


class ButterflyMatrix(Operator):
    orthogonal = True

    # ---- explicit ----
    def generate(self):
        B = torch.ones(1, 1, dtype=self.dtype, device=self.device)
        num_iters = floor(log2(self.d))
        thetas = 2 * pi * torch.rand(num_iters, generator=self.generator, device=self.device, dtype=self.dtype)
        for i in range(num_iters):
            c, s = torch.cos(thetas[i]) * B[:, :self.l], torch.sin(thetas[i]) * B[:, :self.l]
            B = torch.hstack((torch.vstack([c, s]), torch.vstack([-s, c])))
        B = B[:, :self.l]
        if self.d == B.shape[0]:
            return B[:, :self.l].T
        d_eff, d_0 = B.shape[1], B.shape[0]
        to_add = self.d - B.shape[0]
        B = torch.vstack([B, torch.zeros(to_add, B.shape[1], dtype=self.dtype, device=self.device)])
        B = torch.hstack([B, torch.zeros(B.shape[0], to_add, dtype=self.dtype, device=self.device)])
        B[range(d_0, d_0 + to_add), range(d_eff, d_eff + to_add)] = 1.0
        B = B[:, torch.randperm(B.shape[1], generator=self.generator, device=self.device)[:self.l]]
        return B.T

    # ---- implicit ----
    def _resample_implicit(self):
        self.m = floor(log2(self.d))
        self.d0 = 2 ** self.m
        self.to_add = self.d - self.d0
        th = 2 * pi * torch.rand(self.m, generator=self.generator, device=self.device, dtype=self.dtype)
        self.cos, self.sin = torch.cos(th), torch.sin(th)
        self._u = None  # accumulated coefficients, applied once in finalize
        self.d_eff = min(self.d0, self.l)
        if self.to_add == 0:
            self.sel = torch.arange(self.l, device=self.device)
        else:
            self.sel = torch.randperm(self.d_eff + self.to_add, generator=self.generator, device=self.device)[:self.l]

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
        """Columns js of B = R_m (x) ... (x) R_1 as rows. Each is a Kronecker product of m 2-vectors, built as the
        outer product of two half-size factors so the (n, d0) result is written once instead of m times."""
        mid = self.m // 2
        lo = self._half(js, range(mid))
        hi = self._half(js, range(mid, self.m))
        return (hi.unsqueeze(2) * lo.unsqueeze(1)).reshape(js.shape[0], -1)

    def _apply_B(self, u):
        """B @ u for u of length d0 in O(d0 log d0): apply each 2x2 rotation along its own axis."""
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
            Pc[:, :self.d0] = self._bfly_cols(sel)
            return Pc
        is_b = sel < self.d_eff
        Pc[:, :self.d0] = self._bfly_cols(sel.clamp(max=self.d_eff - 1)) * is_b.unsqueeze(1).to(self.dtype)
        off = sel - self.d_eff
        Pc[:, self.d0:] = (off.unsqueeze(1) == self._ar(self.to_add).unsqueeze(0)).to(self.dtype)
        return Pc

    def _perturb_implicit(self, x, h, a, e):
        return torch.add(x, self.chunk(a, e), alpha=h)

    def _accumulate_implicit(self, g, s, a, e):
        # P^T s = B u with u = s scattered at the selected column indices; B u is applied once in finalize().
        # Extra (padding) coordinates are identity rows and are added to g directly.
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
        if self.implicit and self._u is not None:
            g.view(-1)[:self.d0].add_(self._apply_B(self._u))
            self._u = None


class RandomRows(Operator):
    """Rows drawn independently by `draw`. Implicit mode regenerates a chunk from a (base seed + offset)
    generator; with cache=True the chunk made in `perturb` is kept and reused by `accumulate`."""
    draw = None

    def __init__(self, d, l=1, dtype=torch.float64, device="cpu", seed=SEED, implicit=False, cache=True):
        super().__init__(d, l, dtype, device, seed, implicit)
        self.cache = cache
        self.base = 0

    def generate(self):
        return self.draw(self.l, self.d, self.generator, self.dtype, self.device)

    def _resample_implicit(self):
        self.base = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=self.generator, device=self.device).item())

    def chunk(self, a, e):
        gen = torch.Generator(device=self.device).manual_seed(self.base + a)
        return self.draw(e - a, self.d, gen, self.dtype, self.device)

    def _perturb_implicit(self, x, h, a, e):
        Pc = self.chunk(a, e)
        if self.cache:
            self._cached = Pc
        return torch.add(x, Pc, alpha=h)

    def _accumulate_implicit(self, g, s, a, e):
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
    "qr": QRDirections,
    "householder": Householder,
    "permuted_householder": PermutedHouseholder,
    "butterfly": ButterflyMatrix,
    "gaussian": GaussianDirections,
    "spherical": SphericalDirections,
    "rademacher": RademacherDirections,
}

# Named configurations of a family (explicit only). They address the "float32 + re-orthogonalise" suggestion.
VARIANTS = {
    "qr_reorth": ("qr", {"reorth": 1}),                                          # one extra QR pass in `dtype`
    "qr_f32_reorth": ("qr", {"reorth": 1, "gen_dtype": torch.float32}),          # float32 QR twice, cast to `dtype`
}

IMPLICIT_FAMILIES = [n for n, c in FAMILIES.items() if c.supports_implicit]


def make(name, d, l, dtype, device, implicit=False, seed=SEED, **kwargs):
    if name in VARIANTS:
        base, extra = VARIANTS[name]
        kwargs = {**extra, **kwargs}
    else:
        base = name
    return FAMILIES[base](d=d, l=l, dtype=dtype, device=device, seed=seed, implicit=implicit, **kwargs)