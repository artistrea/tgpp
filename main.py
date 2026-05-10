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
    plot_received_signal, plot_time_autocorrelation, ChannelAutocorrelation
)


def main():
    # channel = IdentityChannel(1)
    channel = TGPPChannel(
        100, TGPPChannelModelEnum.UMi
    )

    bs_station = Station(
        pos=Vec3(0., 0., 18.),
    )
    ut_station = Station(
        pos=Vec3(10., 0., 1.5),
        speed_direction=Vec3(1, 0, 0),
        velocity=1.5
    )

    link = Link(
        tx=bs_station,
        rx=ut_station,
        chn=channel,
    )

    impulse_width = 1e-7
    time_samples = int(1e5)
    t = np.linspace(0, impulse_width*5, time_samples)
    s = np.zeros_like(t)
    s[0:len(s)//5] = 1.

    bs_station.signal = BaseBandSignal(3, s, t)

    rng = np.random.default_rng(123)

    link.evaluate_channel(rng)
    link.transmit()

    power = channel.cluster_power

    plot_pdp(
        channel.tau,
        power,
    )

    plot_angular_spectrum(
        channel.doa_azim,
        power,
        angle_name="Azimute",
    )

    plot_angular_spectrum(
        channel.doa_elev,
        power,
        angle_name="Elevação",
    )

    plot_doppler_spectrum(
        channel.doppler,
        power,
        velocity=ut_station.velocity,
    )

    ac = ChannelAutocorrelation(
        power,
        channel.tau,
        channel.doppler,
    )

    plot_time_autocorrelation(
        ac,
        velocity=ut_station.velocity,
        max_doppler=np.max(channel.doppler),
    )

    plot_frequency_autocorrelation(
        ac,
        # rms_delay_spread=results.delay_spread,
    )

    plot_received_signal(
        ut_station.signal.t,
        bs_station.signal.x,
        ut_station.signal.x,
    )

    plt.show()


if __name__ == "__main__":
    main()
