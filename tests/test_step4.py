"""Checks of the Step 4 analyses: spectra, gain limit, capture range, error-point noise."""

import dataclasses

import numpy as np

from pdh_sim import noise, thesis_cavity
from pdh_sim.config import BETA, H0, H1, LOOP, OMEGA_MOD
from pdh_sim.loop import (
    build_discriminator,
    locked_spectra,
    simulate,
    stability_margins,
)
from pdh_sim.pdh import error_signal, optimal_phase


def _psd(f):
    return noise.white_and_flicker_psd(f, H0, H1)


def test_log_bin_preserves_a_flat_spectrum():
    """Averaging a constant PSD in logarithmic bins returns the same constant."""
    f = np.linspace(0, 1e6, 10001)
    fb, sb = noise.log_bin(f, np.full_like(f, 3.0))
    assert np.allclose(sb, 3.0)
    assert np.all(np.diff(fb) > 0)


def test_locked_spectra_match_simulation():
    """The linear-theory spectra of the locked laser agree with the simulation.

    Laser noise and error-point noise are applied together with a linear
    discriminator. Between 10 kHz and 20 MHz the simulated spectrum of the
    transmitted light, averaged in logarithmic bins, must agree with
    locked_spectra on average to 3 %, which tests for a systematic error.
    Individual bins scatter statistically, by up to about 30 % at the low
    end where a bin contains only one or two Welch points, so each bin is
    required to agree to 40 % only.
    """
    cav = thesis_cavity()
    corner = cav.linewidth / 2
    n = 2**19
    s_e = 1e3
    x = noise.generate(n, LOOP.dt, _psd, np.random.default_rng(40))
    e = noise.generate(n, LOOP.dt, lambda f: np.full_like(f, s_e), np.random.default_rng(41))
    d = simulate(x, LOOP, corner, None, error_noise=e)["delta_cav"][2**14:]
    f, s = noise.estimate_psd(d, LOOP.dt, nperseg=2**14)
    fb, sb = noise.log_bin(f, s, per_decade=10)
    theory = locked_spectra(fb, _psd(fb), s_e, LOOP, corner, discrete=True)[1]
    band = (fb > 1e4) & (fb < 2e7)
    ratio = sb[band] / theory[band]
    assert abs(np.mean(ratio) - 1) < 0.03
    assert np.max(np.abs(ratio - 1)) < 0.4


def test_error_point_noise_is_not_suppressed():
    """Within the loop bandwidth, error-point noise reaches the light almost unchanged.

    The transfer function L / (1 + L) is within 10 % of 1 at 10 kHz, in
    contrast to laser noise, which is suppressed there by 1 / |1 + L|^2.
    """
    cav = thesis_cavity()
    f = np.array([1e4])
    _, via_error = locked_spectra(f, 0.0, 1.0, LOOP, cav.linewidth / 2)
    _, via_laser = locked_spectra(f, 1.0, 0.0, LOOP, cav.linewidth / 2)
    assert abs(via_error[0] - 1) < 0.1
    assert via_laser[0] < 0.02


def test_loop_becomes_unstable_near_gain_six():
    """The phase margin reaches zero near G = 6, and the simulated residual grows beyond it."""
    cav = thesis_cavity()
    corner = cav.linewidth / 2
    margin_5 = stability_margins(dataclasses.replace(LOOP, gain=5.0), corner)[1]
    margin_7 = stability_margins(dataclasses.replace(LOOP, gain=7.0), corner)[1]
    assert margin_5 > 0 > margin_7

    disc = build_discriminator(cav, OMEGA_MOD, BETA, optimal_phase(cav, OMEGA_MOD, BETA))
    x = noise.generate(2**17, LOOP.dt, _psd, np.random.default_rng(42))
    rms = {g: np.std(simulate(x, dataclasses.replace(LOOP, gain=g), corner, disc)["delta_cav"][2**13:])
           for g in (3.0, 7.0)}
    assert rms[7.0] > 2 * rms[3.0]


def test_false_lock_point_between_omega_and_two_omega():
    """The error signal crosses zero with the stabilising slope near +32.3 MHz."""
    cav = thesis_cavity()
    phase = optimal_phase(cav, OMEGA_MOD, BETA)
    d = np.linspace(1.02 * OMEGA_MOD, 2 * OMEGA_MOD, 20001)
    e = error_signal(cav, d, OMEGA_MOD, BETA, phase)
    up = np.nonzero((e[:-1] < 0) & (e[1:] >= 0))[0]
    assert up.size >= 1
    assert abs(d[up[0]] - 32.3e6) < 0.1e6


def test_capture_range_is_the_modulation_frequency():
    """Engaged at 19 MHz the lock acquires within 1.5 ms; engaged at 21 MHz it is driven away."""
    cav = thesis_cavity()
    corner = cav.linewidth / 2
    disc = build_discriminator(cav, OMEGA_MOD, BETA, optimal_phase(cav, OMEGA_MOD, BETA),
                               span=4 * OMEGA_MOD, points=400001)
    n = int(1.5e-3 / LOOP.dt)
    x = noise.generate(n, LOOP.dt, _psd, np.random.default_rng(43))
    inside = simulate(19e6 + x, LOOP, corner, disc)["delta_cav"]
    outside = simulate(21e6 + x, LOOP, corner, disc)["delta_cav"]
    assert abs(np.mean(inside[-20000:])) < 5e3
    assert np.mean(outside[-20000:]) > 25e6
