import numpy as np
from enum import Enum, auto

from tgpp.geometry import Vec3
from tgpp.channel.channel import Channel


class TGPPChannelModelEnum(Enum):
    UMi = auto()
    UMa = auto()
    IndoorOpenOffice = auto()


def _get_aoa_azim_std_stats(
    f_c: float,
    propagation_model: TGPPChannelModelEnum,
    los: bool
) -> (float, float):
    """
    Returns (mean, std_variance)
    """
    if propagation_model is TGPPChannelModelEnum.UMi:
        if los:
            mean = -0.08 * np.log10(1 + f_c) + 1.73
            std = 0.14 * np.log10(1 + f_c) + 0.28
        else:
            mean = -0.08 * np.log10(1 + f_c) + 1.81
            std = 0.05 * np.log10(1 + f_c) + 0.3
    elif propagation_model is TGPPChannelModelEnum.UMa:
        if los:
            mean = 1.81
            std = 0.2
        else:
            mean = -0.27 * np.log10(f_c) + 2.08
            std = 0.11
    elif propagation_model is TGPPChannelModelEnum.IndoorOpenOffice:
        if los:
            mean = -0.19 * np.log10(1 + f_c) + 1.781
            std = 0.12 * np.log10(1 + f_c) + 0.119
        else:
            mean = -0.11 * np.log10(1 + f_c) + 1.863
            std = 0.12 * np.log10(1 + f_c) + 0.059
    else:
        raise ValueError(f"propagation_model = {propagation_model} não dá")

    if isinstance(f_c, np.ndarray):
        std = np.zeros_like(f_c) + std
        mean = np.zeros_like(f_c) + mean
    return (mean, std)


def _get_aoa_elev_std_stats(
    f_c: float,
    propagation_model: TGPPChannelModelEnum,
    los: bool
) -> (float, float):
    """
    Returns (mean, std_variance)
    """
    if propagation_model is TGPPChannelModelEnum.UMi:
        if los:
            mean = -0.1 * np.log10(1 + f_c) + 0.73
            std = -0.07 * np.log10(1 + f_c) + 0.34
        else:
            mean = -0.04 * np.log10(1 + f_c) + 0.92
            std = -0.07 * np.log10(1 + f_c) + 0.41
    elif propagation_model is TGPPChannelModelEnum.UMa:
        if los:
            mean = 0.95
            std = 0.16
        else:
            mean = -0.3236 * np.log10(f_c) + 1.512
            std = 0.16
    elif propagation_model is TGPPChannelModelEnum.IndoorOpenOffice:
        if los:
            mean = -0.26 * np.log10(1 + f_c) + 1.44
            std = -0.04 * np.log10(1 + f_c) + 0.264
        else:
            mean = -0.15 * np.log10(1 + f_c) + 1.387
            std = -0.09 * np.log10(1 + f_c) + 0.746
    else:
        raise ValueError(f"propagation_model = {propagation_model} não dá")
    if isinstance(f_c, np.ndarray):
        std = np.zeros_like(f_c) + std
        mean = np.zeros_like(f_c) + mean

    return (mean, std)


def _get_rice_factor_stats(
    propagation_model: TGPPChannelModelEnum,
    los: bool
) -> float:
    if propagation_model is TGPPChannelModelEnum.UMi:
        if los:
            mean = 9
            std = 5
        else:
            mean = 0
            std = 0
    elif propagation_model is TGPPChannelModelEnum.UMa:
        if los:
            mean = 9
            std = 3.5
        else:
            mean = 0
            std = 0
    elif propagation_model is TGPPChannelModelEnum.IndoorOpenOffice:
        if los:
            mean = 7
            std = 4
        else:
            mean = 0
            std = 0
    else:
        raise ValueError(f"propagation_model = {propagation_model} não dá")

    return mean, std


def _get_shadowing_std(
    propagation_model: TGPPChannelModelEnum,
    los: bool
) -> float:
    if propagation_model is TGPPChannelModelEnum.UMi:
        std = 3.
    elif propagation_model is TGPPChannelModelEnum.UMa:
        std = 3.
    elif propagation_model is TGPPChannelModelEnum.IndoorOpenOffice:
        if los:
            std = 6
        else:
            std = 3
    else:
        raise ValueError(f"propagation_model = {propagation_model} não dá")

    return std


def _get_delay_proportionality_factor(
    propagation_model: TGPPChannelModelEnum,
    los: bool
) -> float:
    if propagation_model is TGPPChannelModelEnum.UMi:
        if los:
            factor = 3
        else:
            factor = 2.1
    elif propagation_model is TGPPChannelModelEnum.UMa:
        if los:
            factor = 2.5
        else:
            factor = 2.3
    elif propagation_model is TGPPChannelModelEnum.IndoorOpenOffice:
        if los:
            factor = 3.6
        else:
            factor = 3
    else:
        raise ValueError(f"propagation_model = {propagation_model} não dá")

    return factor


def _get_delay_spread_stats(
    f_c: float,
    propagation_model: TGPPChannelModelEnum,
    los: bool
) -> (float, float):
    """
    Returns (mean, std_variance)
    """
    if propagation_model is TGPPChannelModelEnum.UMi:
        if los:
            mean = -0.24 * np.log10(1 + f_c) - 7.14
            std = 0.38
        else:
            mean = -0.24 * np.log10(1 + f_c) - 6.83
            std = -0.16 * np.log10(1 + f_c) + 0.28
    elif propagation_model is TGPPChannelModelEnum.UMa:
        if los:
            mean = -0.0963 * np.log10(1 + f_c) - 6.955
            std = 0.66
        else:
            mean = -0.204 * np.log10(1 + f_c) - 6.28
            std = 0.39
    elif propagation_model is TGPPChannelModelEnum.IndoorOpenOffice:
        if los:
            mean = -0.1 * np.log10(1 + f_c) - 7.692
            std = 0.18
        else:
            mean = -0.28 * np.log10(1 + f_c) - 7.173
            std = 0.1 * np.log10(1 + f_c) + 0.055
    else:
        raise ValueError(f"propagation_model = {propagation_model} não dá")
    if isinstance(f_c, np.ndarray):
        std = np.zeros_like(f_c) + std
    return (mean, std)


class TGPPChannel(Channel):
    """3GPP's modified channel (without parameters correlation)"""
    def __init__(self, num_paths: int, channel_model: TGPPChannelModelEnum):
        super().__init__(num_paths)
        self.channel_model = channel_model

    def evaluate_los(self, rng):
        d_2d = abs(
            self.tx.pos - self.rx.pos - Vec3(0, 0, self.tx.pos.z-self.rx.pos.z)
        )
        if self.channel_model == TGPPChannelModelEnum.IndoorOpenOffice:
            if d_2d <= 5.:
                return np.ones(1)
            elif d_2d <= 49.:
                return rng.uniform(0, 1, (1)) < np.exp(-(d_2d-5.)/70.8)
            else:
                return rng.uniform(0, 1, (1)) < np.exp(-(d_2d-49.)/211.7)*0.54
        elif self.channel_model == TGPPChannelModelEnum.UMa:
            if d_2d <= 18:
                return np.ones((1))
            else:
                h_ut = self.rx.pos.z
                if h_ut <= 13.:
                    c = 0
                else:
                    c = ((h_ut-13.)/10)**1.5

                term1 = (
                    18 / d_2d
                    + np.exp(-d_2d / 63)
                    * (1 - 18 / d_2d)
                )

                term2 = (
                    1
                    + c
                    * (5 / 4)
                    * (d_2d / 100) ** 3
                    * np.exp(-d_2d / 150)
                )
                return rng.uniform(0, 1, (1)) < term1 * term2
        elif self.channel_model == TGPPChannelModelEnum.UMi:
            if d_2d <= 18:
                return np.ones((1))
            else:
                term = (
                    18 / d_2d
                    + np.exp(-d_2d / 36)
                    * (1 - 18 / d_2d)
                )

                return rng.uniform(0, 1, (1)) < term

        return np.ones(1)

    def evaluate_delays(self, rng):
        delay_spread_mean, delay_spread_std = _get_delay_spread_stats(
            self.tx.signal.f_c, self.channel_model, self.los
        )
        r_tau = _get_delay_proportionality_factor(self.channel_model, self.los)

        sigma_tau_log10 = rng.normal(delay_spread_mean, delay_spread_std)
        # TODO: remove gambiarra
        # lembrar que NAO EH BOM SEPARAR AS COISAS ANTES DA HORA
        # tl;dr abtracao precoce
        self.sigma_tau = np.power(10, sigma_tau_log10)

        # delay samples
        tau_p = rng.exponential(r_tau * self.sigma_tau, self.num_paths)
        # normalized delay samples:
        tau = np.sort(
            tau_p - np.min(
                    tau_p,
                ),
        )

        return tau

    def evaluate_doppler(self, rng):
        arrival_x = (
            np.sin(np.deg2rad(self.doa_elev))
            * np.cos(np.deg2rad(self.doa_azim))
        )
        arrival_y = (
            np.sin(np.deg2rad(self.doa_elev))
            * np.sin(np.deg2rad(self.doa_azim))
        )
        arrival_z = np.cos(np.deg2rad(self.doa_elev))

        speed = self.rx.velocity * np.array([[
            self.rx.speed_direction.x,
            self.rx.speed_direction.y,
            self.rx.speed_direction.z
        ]])

        arrival_vecs = np.transpose([arrival_x, arrival_y, arrival_z])

        lmbda = 3e8 / (self.tx.signal.f_c * 1e9)
        doppler = np.vecdot(arrival_vecs, speed) / lmbda

        return doppler

    def evaluate_powers(self, rng):
        # multipath power
        ksi = rng.normal(
            0, _get_shadowing_std(self.channel_model, self.los), self.num_paths
        )
        r_tau = _get_delay_proportionality_factor(self.channel_model, self.los)
        delay_spread_mean, delay_spread_std = _get_delay_spread_stats(
            self.tx.signal.f_c, self.channel_model, self.los
        )

        cluster_power = np.exp(
            -self.tau * (r_tau - 1) / (r_tau * self.sigma_tau)
        ) * np.power(10, -ksi / 10)

        if self.los:
            Kr_mean, Kr_std = _get_rice_factor_stats(self.channel_model, self.los)
            Kr_dB = rng.normal(Kr_mean, Kr_std)

            K = np.power(10, Kr_dB/10)
            # NOTE: not according to original document
            cluster_power[1:] = cluster_power[1:] / np.sum(cluster_power[1:]) / (K + 1)
            cluster_power[0] = K / (K + 1)
        else:
            cluster_power = cluster_power / np.sum(cluster_power)

        return cluster_power

    def evaluate_doa_azim(self, rng):
        azim_std_mean, azim_std_std = _get_aoa_azim_std_stats(
            self.tx.signal.f_c, self.channel_model, self.los
        )
        azim_std_log = rng.normal(azim_std_mean, azim_std_std)
        azim_std = np.minimum(np.pow(10, azim_std_log), 52)

        azim_p = 1.42 * azim_std * np.sqrt(
            -np.log(self.cluster_power / np.max(self.cluster_power))
        )
        U = rng.choice([-1, 1], size=self.num_paths)
        Y = rng.normal(0, azim_std/7, size=self.num_paths)

        # if dx=1,dy=0, then azim_mean should be 0deg
        # justifying atan2(dy, dx)
        azim_mean = np.rad2deg(np.atan2(
            self.tx.pos.y - self.rx.pos.y,
            self.tx.pos.x - self.rx.pos.x
        ))

        azim = U * azim_p + Y + azim_mean

        if self.los:
            # make first component point to azim_mean
            azim += -azim[0] + azim_mean
        
        azim = np.mod(azim, 360)

        return azim

    def evaluate_doa_elev(self, rng):
        elev_std_mean, elev_std_std = _get_aoa_elev_std_stats(
            self.tx.signal.f_c, self.channel_model, self.los
        )
        elev_std_log = rng.normal(elev_std_mean, elev_std_std)
        elev_std = np.minimum(np.pow(10, elev_std_log), 104.)
        elev_p = -elev_std * np.log(self.cluster_power / np.max(self.cluster_power))

        U = rng.choice([-1, 1], size=self.num_paths)
        Y = rng.normal(0, elev_std/7, size=self.num_paths)
        elev_mean = np.rad2deg(np.atan2(
            np.sqrt(
                (self.tx.pos.y - self.rx.pos.y) ** 2
                + (self.tx.pos.x - self.rx.pos.x) ** 2
            ),
            self.tx.pos.z - self.rx.pos.z
        ))
        elev = U * elev_p + Y + elev_mean

        if self.los:
            elev += - elev[0] + elev_mean

        # NOTE: different from professor's modeling. Considers 3gpp page 38
        # for wrapping zenith angles
        elev = np.mod(elev, 360)

        mask = elev > 180
        elev[mask] = 360 - elev[mask]

        return elev
