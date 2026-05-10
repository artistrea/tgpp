import numpy as np

from tgpp.channel import Channel
from tgpp.station import Station
from tgpp.signal import BaseBandSignal


class Link():
    def __init__(self, tx: Station, rx: Station, chn: Channel):
        self.tx = tx
        self.rx = rx
        self.chn = chn

    def evaluate_channel(self, rng):
        """create the generator: rng=np.random.Generator(103)"""
        self.chn.evaluate(rng, self.tx, self.rx)

    def transmit(self):
        if self.tx.signal:
            # take first drop only as representative
            num_paths = self.chn.num_paths
            tau = self.chn.tau
            a = np.sqrt(self.chn.cluster_power).astype(np.complex128)
            doppler = self.chn.doppler

            tx_time = self.tx.signal.t
            tx_sig = self.tx.signal.x.astype(np.complex128)
            tx_f_c = self.tx.signal.f_c * 1e9
            rx = np.zeros((num_paths, len(tx_time)), dtype=np.complex128)
            step = tx_time[1] - tx_time[0]
            tau_i = np.floor(tau / step).astype(int)

            for i in range(0, num_paths):
                rx[i, tau_i[i]:] = a[i] * np.exp(
                    -2j * np.pi * (
                        (tx_f_c + doppler[i]) * tau[i]
                        - doppler[i] * tx_time[tau_i[i]:]
                    )
                ) * tx_sig[:len(tx_time)-tau_i[i]]

            self.rx.signal = BaseBandSignal(
                tx_f_c,
                np.sum(rx, axis=0),
                tx_time
            )
