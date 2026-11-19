import torch


class LineSearch:

    def __init__(self, gamma0=1.0, c1=1e-4, beta=0.5, gamma_min=1e-12, expand=1.0, gamma_max=float("inf"),
                 n_candidates=1, reset_on_fail=True):
        if not (0 < beta < 1):
            raise ValueError("beta must be in (0, 1)")
        if not (0 < c1 < 1):
            raise ValueError("c1 must be in (0, 1)")
        if expand < 1:
            raise ValueError("expand must be >= 1")
        if gamma_min <= 0 or gamma0 < gamma_min:
            raise ValueError("need 0 < gamma_min <= gamma0")
        if n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        self.gamma0, self.c1, self.beta, self.gamma_min = gamma0, c1, beta, gamma_min
        self.expand, self.gamma_max = expand, gamma_max
        self.n_candidates, self.reset_on_fail = n_candidates, reset_on_fail
        self.n_evals = 0
        self.restart()

    def restart(self):

        self.gamma = self.gamma0
        self.n_evals = 0
        self.done = True
        self.accepted = False
        self.step = 0.0
        self.x_new = self.f_new = None
        self._x = self._g = self._fx = self._gn2 = self._steps = None

    def _fail(self):
        self.done, self.accepted, self.step = True, False, 0.0
        self.x_new = self.f_new = None
        if self.reset_on_fail:
            self.gamma = self.gamma0

    def _next_steps(self):
        steps, t = [], self.gamma
        for _ in range(self.n_candidates):
            if t < self.gamma_min:
                break
            steps.append(t)
            t *= self.beta
        return steps

    def start(self, x, g, fx):
        self._x, self._g, self._fx = x, g, fx
        self.done, self.accepted, self.step = False, False, 0.0
        self.x_new = self.f_new = None
        self._gn2 = (g * g).sum()
        gn2 = self._gn2.item()
        if not (gn2 > 0) or gn2 == float("inf"):
            self._fail()
            return
        self._steps = self._next_steps()
        if not self._steps:
            self._fail()

    def ask(self):
        if self.done:
            raise RuntimeError("line search finished")
        t = torch.tensor(self._steps, dtype=self._x.dtype, device=self._x.device)
        return self._x - t.unsqueeze(1) * self._g

    def tell(self, fvals):
        fvals = fvals.reshape(-1)
        self.n_evals += fvals.numel()
        t = torch.tensor(self._steps, dtype=fvals.dtype, device=fvals.device)
        ok = fvals <= self._fx - self.c1 * t * self._gn2
        hit = ok.nonzero()
        if hit.numel() > 0:
            i = int(hit[0].item())
            self.step = self._steps[i]
            self.x_new = (self._x - self.step * self._g)
            self.f_new = fvals[i].clone()
            self.accepted, self.done = True, True
            self.gamma = min(self.expand * self.step, self.gamma_max)
            return
        self.gamma = self._steps[-1] * self.beta
        self._steps = self._next_steps()
        if not self._steps:
            self._fail()