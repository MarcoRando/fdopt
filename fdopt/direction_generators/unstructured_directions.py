import numpy as np

from fdopt.direction_generators.generator import DirectionGenerator

class GaussianDirections(DirectionGenerator):

    def __call__(self) -> np.ndarray:
        return self.rnd_state.randn(self.l, self.d)

class SphericalDirections(GaussianDirections):
    def __call__(self) -> np.ndarray:
        P = super().__call__().T
        return (P/np.linalg.norm(P, axis=0)).T

class RademacherDirections(DirectionGenerator):
    def __call__(self) -> np.ndarray:
        return self.rnd_state.choice([-1.0, 1.0], size=(self.l, self.d))
