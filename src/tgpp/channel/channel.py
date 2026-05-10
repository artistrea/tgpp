import numpy as np
from abc import ABC

from tgpp.station import Station


class Channel(ABC):
    def __init__(self, num_paths):
        self.num_paths = num_paths
        self.setup()

    def setup(self):
        self.los = None
        self.tau = None
        self.cluster_power = None
        self.doppler = None
        self.doa_azim = None
        self.doa_elev = None
        self.dod_azim = None
        self.dod_elev = None

    def evaluate(self, rng, tx: Station, rx: Station):
        self.setup()
        # yeah, bad practice, so what?
        # NOTE: the channel obviously depends on the damn stations, not
        # sure why I was trying to separate this...
        self.tx = tx
        self.rx = rx

        self.los = self.evaluate_los(rng)
        self.tau = self.evaluate_delays(rng)
        self.cluster_power = self.evaluate_powers(rng)
        self.doa_azim = self.evaluate_doa_azim(rng)
        self.doa_elev = self.evaluate_doa_elev(rng)
        self.doppler = self.evaluate_doppler(rng)

    def evaluate_los(self, rng):
        raise NotImplementedError()

    def evaluate_delays(self, rng):
        raise NotImplementedError()

    def evaluate_doppler(self, rng):
        raise NotImplementedError()

    def evaluate_powers(self, rng):
        raise NotImplementedError()

    def evaluate_doa_azim(self, rng):
        raise NotImplementedError()

    def evaluate_doa_elev(self, rng):
        raise NotImplementedError()
