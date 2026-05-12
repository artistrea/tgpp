from dataclasses import dataclass
import numpy as np


@dataclass
class Vec3():
    x: float
    y: float
    z: float

    def to_list(self):
        return [self.x, self.y, self.z]

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vec3(self.x + other, self.y + other, self.z + other)
        if isinstance(other, Vec3):
            return Vec3(self.x + other.x, self.y + other.x, self.z + other.x)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vec3(self.x * other, self.y * other, self.z * other)

    def __sub__(self, other):
        return self + other * -1

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return self * (1/other)

    def __abs__(self):
        return self.dot(self)

    def dot(self, other):
        return np.sqrt(
            self.x * other.x +
            self.y * other.y +
            self.z * other.z
        )
