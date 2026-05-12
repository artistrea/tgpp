import numpy as np
import matplotlib.pyplot as plt

from tgpp.channel import TGPPChannelModelEnum, IdentityChannel, TGPPChannel
from tgpp.station import Station
from tgpp.geometry import Vec3
from tgpp.link import Link
from tgpp.signal import BaseBandSignal
from tgpp.utils import (
    plot_received_signal, plot_angular_spectrum, plot_arrival_vectors,
    plot_doppler_spectrum, plot_frequency_autocorrelation, plot_pdp,
    plot_received_signal, plot_time_autocorrelation, ChannelAutocorrelation,
    plot_link, plot_los_vs_distance
)


N_TIME_SAMPLES = int(1e5)
N_PATHS = 100
UT_HEIGHT = 1.5
UT_VELOCITY = 0.8
def main():
    umi_link = Link(
        chn=TGPPChannel(N_PATHS, TGPPChannelModelEnum.UMi),
        tx=Station(
            pos=Vec3(0., 0., 10.),
            signal=BaseBandSignal(
                3,
                np.zeros(N_TIME_SAMPLES),
                np.linspace(0, 1e-3, N_TIME_SAMPLES)
            )
        ),
        rx=Station(
            pos=Vec3(100., 0., UT_HEIGHT),
            speed_direction=Vec3(1, 0, 0), # same axis as LoS path
            velocity=UT_VELOCITY
        )
    )

    rng = np.random.default_rng(123)

    umi_link.evaluate_channel(rng)
    umi_link.transmit()

    # power = channel.cluster_power

    plot_link(umi_link)

    plot_los_vs_distance(
        [umi_link],
        600
    )

    plt.show()

if __name__ == "__main__":
    main()
