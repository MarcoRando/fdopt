import torch

GRAD, SEARCH = "grad", "search"


class FFD:
    """Forward finite-difference gradient estimator / optimiser built on the ask-tell protocol.

        ffd = FFD(P, h=1e-6, batch_size=256)
        ffd.reset(x)
        while not ffd.done:
            ffd.tell(f(ffd.ask()))        # first ask() returns x, then batches x + h * P[a:e]
        g = ffd.gradient()

    `estimate(f, x)` runs exactly this loop up to the gradient; `optimize(f, x0, n_iters, gamma)` repeats it with a
    step after each completed estimator. `f` maps a (n, d) batch of points to n values.
    Works with explicit and implicit operators alike (the operator hides the difference).
    x may be (d,) or (1, d); gradient() and the iterate returned by optimize() have the shape of x, while `f`
    always receives 2-D batches (n, d).

    Phases (`phase`):
      GRAD    ask() -> x (only if f(x) is not known yet), then the batches x + h P[a:e]
      SEARCH  (only with `linesearch`) ask() -> the trial points x - t g of the line search
    The gradient is ready (`gradient_ready`) when all l directions have been told; `done` means the whole iteration
    is complete: gradient ready and, if a line search is attached, the search finished. `step()` then moves.

    With a line search, f(x) is not asked again at the next iteration: the accepted trial value (or the old f(x),
    if the search was interrupted and x did not move) is reused. `n_evals` counts the function values told.
    """

    def __init__(self, P, h=1e-6, batch_size=None, norm_const=None, gamma=0.0, x0=None, linesearch=None):
        self.P, self.h, self.gamma, self.ls = P, h, gamma, linesearch
        self.b = P.l if batch_size is None else min(batch_size, P.l)
        self.c = P.norm_const if norm_const is None else norm_const
        self.x = self.g = self.fx = None   # internal x, g are always (1, d)
        self._shape = None
        self.a = 0
        self.phase = GRAD
        self.n_evals = 0
        self.last_step = None
        self.steps = []
        if x0 is not None:
            self.reset(x0.clone())

    # ---- ask / tell -----------------------------------------------------------
    def reset(self, x=None, fx=None):
        """Start a new estimator at x (kept as is, never modified in place) with freshly sampled directions.
        If f(x) is already known pass it as `fx` to skip its evaluation."""
        if x is not None:
            self._shape = x.shape
            self.x = x.reshape(1, -1)
        if self.x is None:
            raise RuntimeError("no point set: pass x to reset() or x0 to the constructor")
        self.P.resample()
        self.g = torch.zeros(self.x.shape, dtype=self.x.dtype, device=self.x.device)  # contiguous (operators use g.view(-1))
        self.a = 0
        self.fx = None if fx is None else fx.reshape(())
        self.phase = GRAD

    @property
    def gradient_ready(self):
        return self.fx is not None and self.a >= self.P.l

    @property
    def done(self):
        if not self.gradient_ready:
            return False
        return self.ls is None or (self.phase == SEARCH and self.ls.done)

    def ask(self):
        if self.x is None:
            raise RuntimeError("call reset(x) first")
        if self.phase == SEARCH:
            return self.ls.ask()
        if self.fx is None:
            return self.x
        if self.gradient_ready:
            raise RuntimeError("estimator complete: call gradient()/step()/reset()")
        return self.P.perturb(self.x, self.h, self.a, min(self.a + self.b, self.P.l))

    def tell(self, fvals):
        fvals = fvals.reshape(-1)
        self.n_evals += fvals.numel()
        if self.phase == SEARCH:
            self.ls.tell(fvals)
            return
        if self.fx is None:
            self.fx = fvals[0].clone()
            return
        e = min(self.a + self.b, self.P.l)
        self.P.accumulate(self.g, (fvals - self.fx) / self.h, self.a, e)
        self.a = e
        if self.a >= self.P.l:
            self.P.finalize(self.g)
            if self.ls is not None:
                self.ls.start(self.x, self.g * self.c, self.fx)
                self.phase = SEARCH

    def gradient(self):
        return (self.g * self.c).reshape(self._shape)

    def step(self):
        """Move and start the next estimator.
        No line search: x <- x - gamma * gradient.
        Line search: x <- accepted trial point (its value becomes the next f(x)); if the search was interrupted x
        stays and the already known f(x) is kept."""
        if not self.done:
            raise RuntimeError("iteration not complete")
        if self.ls is None:
            self.x = self.x - self.gamma * (self.g * self.c)
            self.last_step = self.gamma
            self.reset()
        elif self.ls.accepted:
            self.x, fx, self.last_step = self.ls.x_new, self.ls.f_new, self.ls.step
            self.reset(fx=fx)
        else:
            self.last_step = 0.0
            self.reset(fx=self.fx)

    # ---- drivers ----------------------------------------------------------------
    def estimate(self, f, x=None):
        """One complete gradient estimate at x (or at the current point); a line search is not run."""
        self.reset(x)
        while not self.gradient_ready:
            self.tell(f(self.ask()))
        return self.gradient()

    def optimize(self, f, x0, n_iters, gamma=None, callback=None):
        """Descent with the finite-difference estimator, driven by ask/tell (fixed step `gamma`, or the line search).

        Returns (x, history) where history[k] is f at the k-th iterate (a tensor of length n_iters).
        `callback(k, x, fx)` is called after each iteration's evaluations, before the step.
        Afterwards: `n_evals` (function values used), `steps` (step taken at each iteration, 0 = search interrupted).
        """
        if gamma is not None:
            self.gamma = gamma
        if self.ls is not None:
            self.ls.restart()
        self.n_evals = 0
        self.steps = []
        self.reset(x0.clone())
        hist = []
        for k in range(n_iters):
            while not self.done:
                self.tell(f(self.ask()))
            hist.append(self.fx.detach().clone())
            if callback is not None:
                callback(k, self.x.reshape(self._shape), self.fx)
            self.step()
            self.steps.append(self.last_step)
        return self.x.reshape(self._shape), torch.stack(hist)