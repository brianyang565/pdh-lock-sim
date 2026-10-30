"""Pound-Drever-Hall error signal: analytic and time-domain calculations.

Conventions
-----------
Complex field amplitudes carry the time dependence exp(-i omega t). With
this convention the round-trip recursion used in the time-domain model
reproduces exactly the reflection coefficient F(delta) of ``cavity.py``.

The laser is phase modulated at frequency ``omega_mod`` with depth
``beta``, so the incident field is

    E_in(t) = sqrt(P0) exp[-i (2 pi delta t + beta sin(2 pi omega_mod t))]
            = sqrt(P0) sum_n J_n(beta) exp[-i 2 pi (delta + n omega_mod) t],

where J_n is the Bessel function of the first kind. Sideband n is
therefore reflected with amplitude a_n = J_n(beta) F(delta + n omega_mod).

The reflected power contains a component at the modulation frequency,

    P_ref(t) ⊃ c1 exp(-i 2 pi omega_mod t) + c.c.,   c1 = P0 sum_n a_{n+1} a_n^*.

The photodiode signal is multiplied by a local oscillator
cos(2 pi omega_mod t + phase) and low-pass filtered. The time average of
that product is the error signal,

    eps(delta) = Re[c1 exp(i phase)].

Keeping only the carrier and the first-order sidebands reduces c1 to
J0 J1 times the combination F(0) F*(+) - F*(0) F(-) of Black (2001), up to
complex conjugation, which follows from the opposite sign convention.
Here all sideband orders are kept, so the analytic result can be compared
with the time-domain simulation to numerical precision.

All frequencies are in Hz and all powers are in units of P0.
"""

from __future__ import annotations

import numpy as np
from scipy import signal, special

from .cavity import Cavity


# ----------------------------------------------------------------------
# Analytic (frequency-domain) error signal
# ----------------------------------------------------------------------
def _n_max(beta: float) -> int:
    """Highest sideband order kept. J_n(beta) is below 1e-16 for n > beta + 20."""
    return int(np.ceil(abs(beta))) + 20


def c1_coefficient(cav: Cavity, delta, omega_mod: float, beta: float):
    """Complex amplitude c1 of the reflected-power component at omega_mod.

    Parameters
    ----------
    delta : float or array
        Carrier detuning from resonance, in Hz.
    omega_mod : float
        Modulation frequency (in Hz, not rad/s).
    beta : float
        Phase-modulation depth in rad.
    """
    delta = np.asarray(delta, dtype=float)
    n_max = _n_max(beta)
    # Accumulate c1 = sum_n a_{n+1} a_n^* one sideband pair at a time, so
    # that memory use scales with the number of detunings only.
    a_prev = special.jv(-n_max, beta) * cav.reflection(delta - n_max * omega_mod)
    c1 = np.zeros(delta.shape, dtype=complex)
    for n in range(-n_max + 1, n_max + 1):
        a_n = special.jv(n, beta) * cav.reflection(delta + n * omega_mod)
        c1 += a_n * np.conj(a_prev)
        a_prev = a_n
    return c1


def error_signal(cav: Cavity, delta, omega_mod: float, beta: float, phase: float):
    """Demodulated PDH error signal eps(delta) = Re[c1 exp(i phase)]."""
    return np.real(c1_coefficient(cav, delta, omega_mod, beta) * np.exp(1j * phase))


def optimal_phase(cav: Cavity, omega_mod: float, beta: float, h: float | None = None) -> float:
    """Demodulation phase that maximises the error-signal slope at resonance.

    Near resonance c1 is linear in delta, c1 ≈ c1' delta, so the slope is
    Re[c1' exp(i phase)]. This is largest, and positive, when
    phase = -arg(c1'). The derivative c1' is taken by a central difference
    with step ``h`` (default: 1e-4 of the cavity linewidth).
    """
    h = 1e-4 * cav.linewidth if h is None else h
    dc1 = (
        c1_coefficient(cav, h, omega_mod, beta) - c1_coefficient(cav, -h, omega_mod, beta)
    ) / (2 * h)
    return float(-np.angle(dc1))


def slope_at_resonance(cav: Cavity, omega_mod: float, beta: float, phase: float,
                       h: float | None = None) -> float:
    """Error-signal slope d eps / d delta at delta = 0, in units of P0 per Hz."""
    h = 1e-4 * cav.linewidth if h is None else h
    return float(
        (error_signal(cav, h, omega_mod, beta, phase)
         - error_signal(cav, -h, omega_mod, beta, phase)) / (2 * h)
    )


# ----------------------------------------------------------------------
# Time-domain simulation
# ----------------------------------------------------------------------
def simulate_reflection(cav: Cavity, laser_phase):
    """Reflected field for an arbitrary laser phase history.

    The simulation advances in steps of one cavity round trip, T = 1/FSR,
    and works in a frame rotating at the resonance frequency. At step k
    the circulating field just inside the input mirror is

        a_k = t E_in,k + r^2 a_{k-1},

    and the reflected field is the promptly reflected part of the input
    plus the leakage of the field that has completed one round trip,

        E_ref,k = -r E_in,k + t r a_{k-1}.

    Because light re-enters the input mirror exactly one round trip later,
    this recursion is an exact description of the cavity for any input
    sampled at the round-trip interval. For a monochromatic input it gives
    E_ref = F(delta) E_in, which ``tests/test_pdh.py`` checks.

    Parameters
    ----------
    laser_phase : array
        Optical phase of the incident field at t_k = k T, in rad, measured
        in the rotating frame. The incident field is exp(-i laser_phase),
        normalised to unit power.

    Returns
    -------
    E_ref : complex array
        Reflected field at t_k, in units of sqrt(P0).
    """
    r = cav.r
    t = np.sqrt(1.0 - r**2)
    e_in = np.exp(-1j * np.asarray(laser_phase, dtype=float))
    # IIR filter for a_k = t e_in,k + r^2 a_{k-1}, starting from an empty cavity.
    a = signal.lfilter([t], [1.0, -(r**2)], e_in)
    a_prev = np.concatenate(([0.0], a[:-1]))
    return -r * e_in + t * r * a_prev


def phase_modulated_laser(times, delta, omega_mod: float, beta: float):
    """Optical phase 2 pi ∫delta dt + beta sin(2 pi omega_mod t) on the time grid.

    ``delta`` may be a constant or an array of instantaneous detunings, one
    per time sample, in which case it is integrated with the trapezoidal
    rule.
    """
    times = np.asarray(times, dtype=float)
    delta = np.broadcast_to(np.asarray(delta, dtype=float), times.shape)
    dt = np.diff(times)
    carrier = np.concatenate(([0.0], np.cumsum(np.pi * (delta[1:] + delta[:-1]) * dt)))
    carrier += 2 * np.pi * delta[0] * times[0]
    return carrier + beta * np.sin(2 * np.pi * omega_mod * times)


def demodulate(times, power, omega_mod: float, phase: float, cutoff: float, order: int = 4):
    """Mix the photodiode signal with the local oscillator and low-pass filter it.

    The mixer output P(t) cos(2 pi omega_mod t + phase) contains the error
    signal at DC together with components at omega_mod, 2 omega_mod and
    higher harmonics. A causal Butterworth filter of the given order and
    cutoff frequency removes those components, as the low-pass filter after
    a physical mixer does.
    """
    times = np.asarray(times, dtype=float)
    fs = 1.0 / (times[1] - times[0])
    mixed = power * np.cos(2 * np.pi * omega_mod * times + phase)
    sos = signal.butter(order, cutoff, btype="low", fs=fs, output="sos")
    return signal.sosfilt(sos, mixed)


def lowpass_delay(cutoff: float, order: int = 4) -> float:
    """Low-frequency group delay of the Butterworth filter used in ``demodulate``, in s.

    It is evaluated for the analog prototype, which the digital filter
    matches closely because the cutoff lies far below the sampling rate.
    For a fourth-order filter it equals 2.61 / (2 pi cutoff).
    """
    b, a = signal.butter(order, 2 * np.pi * cutoff, btype="low", analog=True)
    w = 2 * np.pi * cutoff * 1e-4
    _, h1 = signal.freqs(b, a, worN=[w])
    _, h2 = signal.freqs(b, a, worN=[1.001 * w])
    return float(-(np.angle(h2[0]) - np.angle(h1[0])) / (0.001 * w))


def static_error_time_domain(cav: Cavity, delta: float, omega_mod: float, beta: float,
                             phase: float, settle: float = 50.0, periods: int = 20) -> float:
    """Error signal at a fixed detuning, from the time-domain simulation.

    The cavity is driven from empty for ``settle`` photon lifetimes. The
    field amplitude decays with time constant 2 tau, where tau is the
    photon (energy) lifetime, so after 50 lifetimes the transient field is
    reduced by exp(-25), about 1e-11, relative to its initial value. With
    20 lifetimes the residual, about 5e-5 in field, would limit the
    agreement with the analytic result to about 1e-6. The mixer
    output is then averaged over an integer number of modulation periods,
    which removes every harmonic of omega_mod exactly, so that this
    function isolates the physics of the cavity from the choice of
    low-pass filter.

    The modulation period must be an integer number of round trips; for
    the thesis cavity FSR / omega_mod = 1.5 GHz / 20 MHz = 75.
    """
    steps_per_period = cav.fsr / omega_mod
    if not np.isclose(steps_per_period, round(steps_per_period), rtol=0, atol=1e-9):
        raise ValueError("FSR / omega_mod must be an integer for exact period averaging")
    steps_per_period = int(round(steps_per_period))

    n_settle = int(np.ceil(settle * cav.photon_lifetime * cav.fsr / steps_per_period))
    n_total = (n_settle + periods) * steps_per_period
    times = np.arange(n_total) * cav.round_trip_time

    e_ref = simulate_reflection(cav, phase_modulated_laser(times, delta, omega_mod, beta))
    power = np.abs(e_ref) ** 2
    mixed = power * np.cos(2 * np.pi * omega_mod * times + phase)
    return float(np.mean(mixed[-periods * steps_per_period:]))


def frequency_modulation_response(cav: Cavity, f_fm: float, omega_mod: float, beta: float,
                                  phase: float, amplitude: float = 5e3,
                                  settle: float = 10e-6, min_duration: float = 20e-6) -> complex:
    """Response of the error signal to a sinusoidal laser-frequency modulation.

    The laser detuning is delta(t) = amplitude * sin(2 pi f_fm t), which is
    applied to the full round-trip simulation together with the PDH phase
    modulation. The mixer output is averaged over each modulation period,
    which removes every harmonic of omega_mod exactly, and the resulting
    series is demodulated at f_fm.

    The result is normalised to the static response slope * amplitude, so
    that it equals 1 for a cavity that responds instantaneously. For a
    high-finesse cavity in the fast-modulation regime it is expected to
    follow a single pole at the half-width, 1 / (1 + i f_fm / (linewidth / 2)).

    Both omega_mod / f_fm and FSR / omega_mod must be integers, so that the
    record contains whole periods of both modulations.
    """
    n_per = cav.fsr / omega_mod
    k = omega_mod / f_fm
    if not (np.isclose(n_per, round(n_per), atol=1e-9) and np.isclose(k, round(k), atol=1e-9)):
        raise ValueError("FSR / omega_mod and omega_mod / f_fm must be integers")
    n_per, k = int(round(n_per)), int(round(k))
    dt = cav.round_trip_time

    cycles = max(4, int(np.ceil(min_duration * f_fm)))
    n_settle = int(np.ceil(settle * omega_mod))
    n_blocks = n_settle + cycles * k
    t = np.arange(n_blocks * n_per) * dt

    laser = -(amplitude / f_fm) * np.cos(2 * np.pi * f_fm * t) + beta * np.sin(2 * np.pi * omega_mod * t)
    power = np.abs(simulate_reflection(cav, laser)) ** 2
    mixed = power * np.cos(2 * np.pi * omega_mod * t + phase)

    blocks = mixed.reshape(n_blocks, n_per).mean(axis=1)[n_settle:]
    t_blocks = t.reshape(n_blocks, n_per)[n_settle:, 0] + 0.5 * (n_per - 1) * dt
    z = 2 * np.mean(blocks * np.exp(-2j * np.pi * f_fm * t_blocks))

    # Block averaging multiplies a component at f_fm by this real factor
    # (timestamps are taken at the block centres, so there is no phase shift).
    box = np.sin(np.pi * f_fm * n_per * dt) / (n_per * np.sin(np.pi * f_fm * dt))
    slope = slope_at_resonance(cav, omega_mod, beta, phase)
    # delta(t) = amplitude * sin(...) has complex amplitude -i * amplitude.
    return complex(z / (-1j * amplitude * slope * box))


def shot_noise_limit(cav: Cavity, carrier_power: float, wavelength: float) -> float:
    """Shot-noise-limited frequency sensitivity of the PDH error signal, in Hz/sqrt(Hz).

    Black [Am. J. Phys. 69, 79 (2001)] gives

        S^(1/2) = sqrt(h c^3) / (8 L F sqrt(lambda P_c)),

    where L is the cavity length, F the finesse and P_c the carrier power.
    It is the smallest frequency noise that photon shot noise on the PDH
    photodiode can imitate; noise entering the error signal above this
    level is technical.
    """
    h = 6.62607015e-34
    c = 299792458.0
    length = c / (2 * cav.fsr)
    return float(np.sqrt(h * c**3) / (8 * length * cav.finesse * np.sqrt(wavelength * carrier_power)))
