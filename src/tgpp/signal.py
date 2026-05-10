import numpy as np
from dataclasses import dataclass


@dataclass
class BaseBandSignal():
    """
    Defines a baseband signal x(t) in both axis (sample value and time)
    for transmission at f_c [GHz]
    """
    f_c: float
    x: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        if self.t.shape != self.x.shape:
            raise ValueError("x(t) and its associated t must have same shape")

        if self.x.ndim != 1:
            raise ValueError("x(t) must be one dimensional")

        if self.f_c <= 0.:
            raise ValueError("f_c must be > 0")
