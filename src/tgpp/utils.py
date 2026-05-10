import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PDP — Power Delay Profile
# ============================================================


def plot_pdp(
    delay,
    power,
    *,
    ax=None,
    delay_unit="us",
    title="Perfil de atraso de potência",
    highlight_los=True,
    scenario=None
):
    """
    delay: seconds
    power: linear power
    """

    scale = {
        "s": 1,
        "ms": 1e3,
        "us": 1e6,
        "ns": 1e9,
    }[delay_unit]

    delay_scaled = delay * scale

    if ax is None:
        fig, ax = plt.subplots()

    ax.stem(delay_scaled, power, markerfmt="^k", linefmt="k")

    if highlight_los and len(delay_scaled) > 0:
        ax.stem(
            [delay_scaled[0]],
            [power[0]],
            markerfmt="^b",
            linefmt="b",
        )

    ax.set_yscale("log")
    ax.grid(True)
    if scenario is not None:
        title += f" - {scenario}"
    ax.set_title(title)
    ax.set_xlabel(rf"Atraso [{delay_unit}]")
    ax.set_ylabel("Potência")

    return ax


# ============================================================
# Angular Spectrum
# ============================================================


def plot_angular_spectrum(
    angles,
    power,
    *,
    angle_name="Azimute",
    polar=True,
    cartesian=True,
    fig=None,
    scenario=None
):
    """
    angles: degrees
    power: linear power
    """

    if polar and cartesian:
        if fig is None:
            fig = plt.figure(figsize=(12, 5))

        ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
        ax_cart = fig.add_subplot(1, 2, 2)

    elif polar:
        if fig is None:
            fig = plt.figure()

        ax_polar = fig.add_subplot(111, projection="polar")
        ax_cart = None

    else:
        if fig is None:
            fig = plt.figure()

        ax_cart = fig.add_subplot(111)
        ax_polar = None

    if ax_polar is not None:
        for i in range(len(angles)):
            ax_polar.plot(
                [np.deg2rad(angles[i]), np.deg2rad(angles[i])],
                [1e-9, power[i]],
                color="black",
                linewidth=1,
            )

        markerline, *_ = ax_polar.stem(
            np.deg2rad(angles),
            power,
            markerfmt="o",
            linefmt="k",
        )

        markerline.set(markerfacecolor="none")

        ax_polar.set_rscale("log")
        ax_polar.set_rlabel_position(-22.5)
        ax_polar.grid(True)

    if ax_cart is not None:
        ax_cart.stem(
            angles,
            power,
            markerfmt="^k",
            linefmt="k",
        )

        ax_cart.set_yscale("log")
        ax_cart.grid(True)

        ax_cart.set_xlabel(rf"{angle_name} [$^\circ$]")
        ax_cart.set_ylabel("Potência")

    fig.suptitle(f"Espectro angular de potência ({scenario} - {angle_name.lower()})")

    return fig


# ============================================================
# Arrival Vectors
# ============================================================


def spherical_to_cartesian(
    azimuth,
    elevation,
    radius,
):
    x = np.sin(elevation) * np.cos(azimuth) * radius
    y = np.sin(elevation) * np.sin(azimuth) * radius
    z = np.cos(elevation) * radius

    return x, y, z



def plot_arrival_vectors(
    azimuth,
    elevation,
    power,
    *,
    ax=None,
    title="Vetores de chegada multipercurso",
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    x, y, z = spherical_to_cartesian(
        azimuth,
        elevation,
        power,
    )

    ax.quiver(
        0,
        0,
        0,
        x,
        y,
        z,
        arrow_length_ratio=0.1,
    )

    mx = np.max(power)

    ax.set_xlim([-mx, mx])
    ax.set_ylim([-mx, mx])
    ax.set_zlim([-mx, mx])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)

    return ax


# ============================================================
# Doppler Spectrum
# ============================================================


def plot_doppler_spectrum(
    doppler,
    power,
    *,
    ax=None,
    title="Espectro Doppler",
    velocity=None,
    highlight_los=True,
    scenario=None
):
    if ax is None:
        fig, ax = plt.subplots()

    ax.stem(
        doppler,
        power,
        markerfmt="^k",
        linefmt="k",
    )

    if highlight_los and len(doppler) > 0:
        ax.stem(
            [doppler[0]],
            [power[0]],
            markerfmt="^b",
            linefmt="b",
        )

    ax.set_yscale("log")
    ax.grid(True)

    if velocity is not None:
        title = f"{title} — v = {velocity} m/s"

    if scenario is not None:
        title += f" - {scenario}"

    ax.set_title(title)
    ax.set_xlabel("Desvio Doppler [Hz]")
    ax.set_ylabel("Potência")

    return ax


# ============================================================
# Signal Plot
# ============================================================


def plot_received_signal(
    t,
    tx_signal,
    rx_signal,
    *,
    ax=None,
    xlim=None,
    title="Sinais transmitido e recebido",
):
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(
        t,
        np.abs(tx_signal),
        color="blue",
        label="Transmitido",
    )

    ax.plot(
        t,
        np.abs(rx_signal),
        color="red",
        label="Recebido",
    )

    ax.grid(True)
    ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Tempo [s]")
    ax.set_ylabel(r"$|\tilde{r}(t)|$")

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim((t[0], t[-1]))

    return ax


# ============================================================
# Channel Autocorrelation
# ============================================================


class ChannelAutocorrelation:
    def __init__(self, power, delay, doppler):
        self.power = np.asarray(power)
        self.delay = np.asarray(delay)
        self.doppler = np.asarray(doppler)

        self.omega_c = np.sum(power)

    def evaluate(self, kappa, omega):
        kappa = np.reshape(kappa, (1, len(kappa)))
        omega = np.reshape(omega, (1, len(omega)))

        return np.sum(
            self.power[:, np.newaxis]
            * np.exp(-2j * np.pi * self.delay[:, np.newaxis] * kappa)
            * np.exp(2j * np.pi * self.doppler[:, np.newaxis] * omega),
            axis=0,
        ) / self.omega_c


# ============================================================
# Time Autocorrelation
# ============================================================


def coherence_time(
    autocorr,
    time_lags,
    threshold,
):
    idx = np.where(np.abs(autocorr) <= threshold)[0]

    if len(idx) == 0:
        return None

    return time_lags[idx[0]]



def plot_time_autocorrelation(
    autocorr_builder,
    *,
    threshold_1=0.95,
    threshold_2=0.90,
    t_min=1e-6,
    t_max=1,
    n_points=10000,
    ax=None,
    velocity=None,
    max_doppler=None,
    scenario=None
):
    t = np.logspace(
        np.log10(t_min),
        np.log10(t_max),
        n_points,
    )

    rho = np.abs(
        autocorr_builder.evaluate(
            np.zeros_like(t),
            t,
        )
    )

    tc_1 = coherence_time(rho, t, threshold_1)
    tc_2 = coherence_time(rho, t, threshold_2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t, rho)

    if tc_1 is not None:
        ax.vlines(tc_1, -0.1, 1.1, linestyle=":", color="k")
        ax.hlines(threshold_1, t[0], t[-1], linestyle=":", color="k")

    if tc_2 is not None:
        ax.vlines(tc_2, -0.1, 1.1, linestyle=":", color="k")
        ax.hlines(threshold_2, t[0], t[-1], linestyle=":", color="k")

    title = ["Autocorr. temporal"]

    if scenario is not None:
        title.append(scenario)

    if tc_1 is not None:
        title.append(rf"$T_C({threshold_1}) = {tc_1 * 1e3:.2g}$ ms")

    if tc_2 is not None:
        title.append(rf"$T_C({threshold_2}) = {tc_2 * 1e3:.2g}$ ms")

    if max_doppler is not None:
        title.append(rf"max$(\nu_n)$ = {max_doppler:.0f} Hz")

    if velocity is not None:
        title.append(rf"$v_{{rx}} = {velocity}$ m/s")

    ax.set_title(", ".join(title))

    ax.set_xscale("log")

    ax.set_xlim((t_min, t_max))
    ax.set_ylim((-0.1, 1.1))

    ax.grid(True)

    ax.set_xlabel("Desvio temporal [s]")
    ax.set_ylabel(r"$|\rho_{TT}(0,\sigma)|$")

    return ax


# ============================================================
# Frequency Autocorrelation
# ============================================================


def coherence_bandwidth(
    autocorr,
    frequency_lags,
    threshold,
):
    idx = np.where(np.abs(autocorr) <= threshold)[0]

    if len(idx) == 0:
        return None

    return frequency_lags[idx[0]]



def plot_frequency_autocorrelation(
    autocorr_builder,
    *,
    threshold_1=0.95,
    threshold_2=0.90,
    f_min=1,
    f_max=1e10,
    n_points=10000,
    ax=None,
    rms_delay_spread=None,
    scenario=None
):
    f = np.logspace(
        np.log10(f_min),
        np.log10(f_max),
        n_points,
    )

    rho = np.abs(
        autocorr_builder.evaluate(
            f,
            np.zeros_like(f),
        )
    )

    bc_1 = coherence_bandwidth(rho, f, threshold_1)
    bc_2 = coherence_bandwidth(rho, f, threshold_2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(f, rho)

    if bc_1 is not None:
        ax.vlines(bc_1, -0.1, 1.1, linestyle=":", color="k")
        ax.hlines(threshold_1, f[0], f[-1], linestyle=":", color="k")

    if bc_2 is not None:
        ax.vlines(bc_2, -0.1, 1.1, linestyle=":", color="k")
        ax.hlines(threshold_2, f[0], f[-1], linestyle=":", color="k")

    title = ["Autocorr. em frequência"]

    if scenario is not None:
        title.append(scenario)

    if bc_1 is not None:
        title.append(rf"$B_C({threshold_1}) = {bc_1 / 1e6:.2g}$ MHz")

    if bc_2 is not None:
        title.append(rf"$B_C({threshold_2}) = {bc_2 / 1e6:.2g}$ MHz")

    if rms_delay_spread is not None:
        title.append(rf"$\sigma_\tau = {rms_delay_spread * 1e6:.2g}\, \mu s$")

    ax.set_title(", ".join(title))

    ax.set_xscale("log")

    ax.set_xlim((f_min, f_max))
    ax.set_ylim((-0.1, 1.1))

    ax.grid(True)

    ax.set_xlabel("Desvio de frequência [Hz]")
    ax.set_ylabel(r"$|\rho_{TT}(\kappa,0)|$")

    return ax

def compute_delay_spread(tau, cluster_pow):
    omega_c = np.sum(cluster_pow)

    tau_mean = np.sum(
        tau * cluster_pow
    ) / omega_c

    sigma_tau = np.sqrt(
        np.sum(
            cluster_pow * (tau - tau_mean)**2
        ) / omega_c
    )

    return sigma_tau


# ============================================================
# Example Usage
# ============================================================


"""

plot_pdp(
    results.clusters_delay,
    results.clusters_power,
)

plot_angular_spectrum(
    results.clusters_azimuth,
    results.clusters_power,
    angle_name="Azimute",
)

plot_angular_spectrum(
    results.clusters_elevation,
    results.clusters_power,
    angle_name="Elevação",
)

plot_arrival_vectors(
    results.clusters_azimuth,
    results.clusters_elevation,
    results.clusters_power,
)

plot_doppler_spectrum(
    results.clusters_doppler,
    results.clusters_power,
    velocity=5,
)

plot_received_signal(
    results.t_signal,
    results.tx_signal,
    results.rx_signal,
)

ac = ChannelAutocorrelation(
    results.clusters_power,
    results.clusters_delay,
    results.clusters_doppler,
)

plot_time_autocorrelation(
    ac,
    velocity=5,
    max_doppler=np.max(results.clusters_doppler),
)

plot_frequency_autocorrelation(
    ac,
    rms_delay_spread=results.delay_spread,
)

plt.show()

"""

