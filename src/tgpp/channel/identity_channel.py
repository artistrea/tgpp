from tgpp.channel.channel import Channel

import numpy as np


class IdentityChannel(Channel):
    def __init__(self, num_paths):
        super().__init__(num_paths)

        if self.num_paths != 1:
            raise ValueError("Identity channel must have a single path")

    def evaluate_los(self, rng):
        return np.ones(1)

    def evaluate_delays(self, rng):
        return np.zeros((self.num_paths))

    def evaluate_doppler(self, rng):
        return np.zeros((self.num_paths))

    def evaluate_powers(self, rng):
        return np.ones((self.num_paths))

    def evaluate_doa_azim(self, rng):
        return np.zeros((self.num_paths))

    def evaluate_doa_elev(self, rng):
        return np.zeros((self.num_paths))

    def evaluate_dod_azim(self, rng):
        return np.zeros((self.num_paths))

    def evaluate_dod_elev(self, rng):
        return np.zeros((self.num_paths))
