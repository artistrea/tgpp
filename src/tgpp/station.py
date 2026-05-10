from dataclasses import dataclass, field
import typing

from tgpp.signal import BaseBandSignal
from tgpp.geometry import Vec3


@dataclass
class Station():
    pos: Vec3
    speed_direction: Vec3 = field(default_factory=lambda: Vec3(0, 0, 1))
    velocity: float = 0.
    signal: typing.Optional[BaseBandSignal] = None

    def __post_init__(self):
        self.speed_direction = self.speed_direction / abs(self.speed_direction)
