import numpy as np
from fdopt.direction_generators.generator import DirectionGenerator

class CoordinateDirections(DirectionGenerator):

    def __call__(self) -> np.ndarray:
        I = np.zeros(self.l, self.d, dtype=self.dtype)
        I[range(self.l), self.rnd_state.choice(self.d, size=self.l, replace=False)] = 1.0
        return I
