"""Closed-loop simulation of the PDH lock.

Signal chain
------------
The laser frequency relative to the cavity resonance is

    delta = nu_free + nu_act,

where nu_free is the free-running frequency noise and nu_act is the
correction applied by the actuator. Around the loop:

1. The cavity responds to delta as a single-pole low-pass filter with its
   corner at the half-width, linewidth / 2 (375 kHz for the thesis cavity).
   ``pdh.frequency_modulation_response`` verifies this against the full
   optical simulation.
2. The filtered detuning is converted into an error signal through the
   static PDH curve. The curve is divided by its slope at resonance, so
   the error is expressed in Hz and equals the detuning for small
   excursions. Beyond the linear region the full nonlinear curve is used,
   which allows the loop to lose lock.
3. The error signal reaches the actuator after a fixed delay (electronics,
   cables and the laser headboard).
4. A proportional-integral (PI) controller computes the correction
   u = -G (e + 2 pi f_int ∫e dt). The dimensionless proportional gain G is
   the loop gain at low frequency, above the integrator corner f_int.
5. The diode-current actuator follows u with a single-pole response of
   bandwidth f_act.

Modelling the cavity as a low-pass filter followed by the static curve is
valid while the laser stays within the central slope and the modulation
frequencies of interest are well below the PDH modulation frequency, as
the check against the optical simulation shows. It avoids stepping the
optical field every 0.67 ns, so that milliseconds of locked operation can
be simulated in seconds.

Units: frequencies in Hz, times in s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cavity import Cavity
from .pdh import error_signal, slope_at_resonance


@dataclass(frozen=True)
class LoopParams:
    """Parameters of the feedback loop.

    gain : float
        Proportional loop gain G (dimensionless).
    f_int : float
        Integrator corner frequency of the PI controller, Hz.
    f_act : float
        Bandwidth of the current actuator, Hz.
    delay : float
        Total loop delay, s.
    dt : float
        Simulation time step, s.
    """

    gain: float
    f_int: float
    f_act: float
    delay: float
    dt: float = 10e-9


# ----------------------------------------------------------------------
# Discriminator
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Discriminator:
    """The static PDH curve on a uniform grid, in units of Hz of detuning.

    ``values[i]`` is eps(x_i) / slope, where x_i = x0 + i * dx. Outside the
    table the edge values are used.
    """

    x0: float
    dx: float
    values: np.ndarray

    def __call__(self, delta):
        return np.interp(delta, self.x0 + self.dx * np.arange(self.values.size), self.values)


def build_discriminator(cav: Cavity, omega_mod: float, beta: float, phase: float,
                        span: float | None = None, points: int = 200001) -> Discriminator:
    """Tabulate eps(delta) / slope over +-span (default +-1.5 omega_mod)."""
    span = 1.5 * omega_mod if span is None else span
    x = np.linspace(-span, span, points)
    slope = slope_at_resonance(cav, omega_mod, beta, phase)
    values = error_signal(cav, x, omega_mod, beta, phase) / slope
    return Discriminator(x0=float(x[0]), dx=float(x[1] - x[0]), values=values)


# ----------------------------------------------------------------------
# Time-domain loop
# ----------------------------------------------------------------------
def simulate(nu_free, params: LoopParams, cavity_corner: float,
             discriminator: Discriminator | None = None, engage: int = 0,
             error_noise=None):
    """Run the loop for the given free-running frequency noise.

    Parameters
    ----------
    nu_free : array
        Free-running laser frequency relative to the cavity resonance, Hz,
        sampled at params.dt. A constant offset may be included.
    cavity_corner : float
        Corner frequency of the cavity low-pass, normally linewidth / 2.
    discriminator : Discriminator, optional
        Static error curve. If omitted, the error equals the filtered
        detuning (a perfectly linear discriminator).
    engage : int
        Sample index at which the controller is switched on. Before it the
        actuator output is zero.
    error_noise : array, optional
        Noise added to the error signal, in Hz of equivalent detuning, one
        value per sample. It represents noise that enters at the detector or
        in the electronics before the controller, such as photodetector and
        amplifier noise or residual amplitude modulation. The loop cannot
        distinguish it from a real frequency error.

    Returns
    -------
    dict with arrays ``delta`` (laser minus cavity), ``delta_cav`` (after
    the cavity low-pass), ``error`` and ``nu_act``.

    Notes
    -----
    Each step updates, in order: the laser detuning, the cavity filter,
    the error, the delay line, the controller and the actuator. The
    actuator output computed at step k is applied at step k + 1, which adds
    one time step to the loop delay; ``open_loop_gain(..., discrete=True)``
    includes it.
    """
    nu_free = np.asarray(nu_free, dtype=float)
    n = nu_free.size
    dt = params.dt
    a_c = 1.0 - np.exp(-2 * np.pi * cavity_corner * dt)
    a_a = 1.0 - np.exp(-2 * np.pi * params.f_act * dt)
    n_delay = int(round(params.delay / dt))
    kp = params.gain
    ki = params.gain * 2 * np.pi * params.f_int * dt

    if discriminator is None:
        x0, inv_dx, table, n_tab = 0.0, 0.0, None, 0
    else:
        x0, inv_dx = discriminator.x0, 1.0 / discriminator.dx
        table = discriminator.values.tolist()
        n_tab = len(table)

    delta = np.empty(n)
    delta_cav = np.empty(n)
    error = np.empty(n)
    nu_act = np.empty(n)

    ring = [0.0] * (n_delay + 1)
    act = 0.0
    dc = 0.0
    integ = 0.0
    free = nu_free.tolist()
    noise_e = [0.0] * n if error_noise is None else np.asarray(error_noise, dtype=float).tolist()

    for k in range(n):
        d = free[k] + act
        dc += a_c * (d - dc)

        if table is None:
            e = dc
        else:
            pos = (dc - x0) * inv_dx
            if pos <= 0.0:
                e = table[0]
            elif pos >= n_tab - 1:
                e = table[-1]
            else:
                i = int(pos)
                frac = pos - i
                e = table[i] + frac * (table[i + 1] - table[i])
        e += noise_e[k]

        ring[k % (n_delay + 1)] = e
        e_d = ring[(k - n_delay) % (n_delay + 1)]

        delta[k] = d
        delta_cav[k] = dc
        error[k] = e
        nu_act[k] = act

        if k >= engage:
            integ += ki * e_d
            u = -(kp * e_d + integ)
        else:
            u = 0.0
        act += a_a * (u - act)

    return {"delta": delta, "delta_cav": delta_cav, "error": error, "nu_act": nu_act}


# ----------------------------------------------------------------------
# Linear analysis
# ----------------------------------------------------------------------
def _loop_components(f, params: LoopParams, cavity_corner: float, discrete: bool):
    """Cavity response H_c(f) and the rest of the loop K(f), with L = H_c K.

    K contains the delay, the PI controller and the actuator. With
    ``discrete=True`` the exact z-domain responses of ``simulate`` are
    returned, including the extra step of delay noted there.
    """
    f = np.asarray(f, dtype=float)
    p = params
    with np.errstate(divide="ignore", invalid="ignore"):  # the integrator diverges at f = 0
        if not discrete:
            s = 2j * np.pi * f
            h_c = 1 / (1 + s / (2 * np.pi * cavity_corner))
            k = (p.gain * (1 + 2 * np.pi * p.f_int / s)
                 / (1 + s / (2 * np.pi * p.f_act)) * np.exp(-s * p.delay))
            return h_c, k
        dt = p.dt
        zi = np.exp(-2j * np.pi * f * dt)  # z^-1
        a_c = 1.0 - np.exp(-2 * np.pi * cavity_corner * dt)
        a_a = 1.0 - np.exp(-2 * np.pi * p.f_act * dt)
        n_delay = int(round(p.delay / dt))
        h_c = a_c / (1 - (1 - a_c) * zi)
        pi_ctrl = p.gain * (1 + 2 * np.pi * p.f_int * dt / (1 - zi))
        act = zi * a_a / (1 - (1 - a_a) * zi)
        return h_c, zi**n_delay * pi_ctrl * act


def open_loop_gain(f, params: LoopParams, cavity_corner: float, discrete: bool = False):
    """Open-loop gain L(f); the laser noise is suppressed by 1 / (1 + L).

    The continuous form is

        L = G (1 + f_int / (i f)) / (1 + i f / f_c) / (1 + i f / f_act) * exp(-i 2 pi f tau).

    With ``discrete=True`` the exact z-domain response of ``simulate`` is
    returned instead, which differs from the continuous form only close to
    the sampling rate.
    """
    h_c, k = _loop_components(f, params, cavity_corner, discrete)
    return h_c * k


def locked_spectra(f, s_free, s_error, params: LoopParams, cavity_corner: float,
                   discrete: bool = False):
    """Linear-theory spectra of the locked laser, in Hz^2/Hz.

    With laser noise nu_free and error-point noise n, the loop equations
    delta = nu_free - K (H_c delta + n) and delta_cav = H_c delta give

        delta     = (nu_free - K n) / (1 + L),
        delta_cav = H_c nu_free / (1 + L) - L n / (1 + L).

    Laser noise is therefore suppressed by 1 / |1 + L|, whereas error-point
    noise is transferred to the light by L / (1 + L), which is close to 1
    at all frequencies where the loop has gain.

    Parameters
    ----------
    s_free, s_error : arrays or floats
        One-sided PSDs of the free-running laser noise and of the
        error-point noise (in Hz^2/Hz of equivalent detuning).

    Returns
    -------
    (s_delta, s_delta_cav) : PSDs of the laser detuning and of the
    detuning seen through the cavity, i.e. the frequency noise carried by
    the cavity-transmitted light relative to the cavity resonance.
    """
    h_c, k = _loop_components(f, params, cavity_corner, discrete)
    L = h_c * k
    with np.errstate(divide="ignore", invalid="ignore"):
        s_delta = (np.abs(1 / (1 + L)) ** 2 * s_free
                   + np.abs(k / (1 + L)) ** 2 * s_error)
        s_cav = (np.abs(h_c / (1 + L)) ** 2 * s_free
                 + np.abs(L / (1 + L)) ** 2 * s_error)
    return s_delta, s_cav


def stability_margins(params: LoopParams, cavity_corner: float,
                      f_min: float = 1e2, f_max: float = 1e9, points: int = 200001):
    """Unity-gain frequency and phase margin of the continuous open-loop gain.

    The unity-gain frequency is the highest frequency at which |L| crosses
    1, and the phase margin is 180 degrees plus the phase of L there. A
    positive margin is required for stability; values of 30 to 60 degrees
    are typical of a well-damped loop.
    """
    f = np.logspace(np.log10(f_min), np.log10(f_max), points)
    L = open_loop_gain(f, params, cavity_corner)
    above = np.abs(L) >= 1
    if not above.any():
        raise ValueError("|L| < 1 everywhere: the loop has no gain")
    i = np.nonzero(above)[0][-1]
    f_u = float(f[i])
    phase = np.unwrap(np.angle(L))[i]
    return f_u, float(np.degrees(phase) + 180.0)
