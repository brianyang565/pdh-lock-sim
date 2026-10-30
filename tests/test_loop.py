"""Checks of the noise model and the closed-loop simulation.

The loop replaces the optical simulation by a reduced model (cavity
low-pass followed by the static error curve), so the first check verifies
that model against the full optical simulation. The loop checks then
compare the simulated noise suppression with the analytic open-loop gain.
"""

import numpy as np

from pdh_sim import noise, thesis_cavity
from pdh_sim.config import BETA, H0, H1, LOOP, OMEGA_MOD
from pdh_sim.loop import (
    build_discriminator,
    open_loop_gain,
    simulate,
    stability_margins,
)
from pdh_sim.pdh import frequency_modulation_response, optimal_phase

N = 2**18          # 2.6 ms at 10 ns steps
SKIP = 2**14       # discard the first 164 us, while the loop settles


def _psd(f):
    return noise.white_and_flicker_psd(f, H0, H1)


def _disc(cav):
    return build_discriminator(cav, OMEGA_MOD, BETA, optimal_phase(cav, OMEGA_MOD, BETA))


def test_cavity_responds_as_single_pole():
    """In the optical simulation, frequency noise reaches the error signal through a pole at linewidth / 2.

    This justifies the reduced cavity model used in the loop. Agreement is
    0.5 % up to 1 MHz; at 4 MHz, a fifth of the modulation frequency,
    corrections of order f / Omega raise the difference to about 1 %.
    """
    cav = thesis_cavity()
    phase = optimal_phase(cav, OMEGA_MOD, BETA)
    corner = cav.linewidth / 2
    for f_fm, tol in ((100e3, 5e-3), (500e3, 5e-3), (1e6, 5e-3), (4e6, 2e-2)):
        measured = frequency_modulation_response(cav, f_fm, OMEGA_MOD, BETA, phase)
        expected = 1 / (1 + 1j * f_fm / corner)
        assert abs(measured / expected - 1) < tol


def test_noise_generator_reproduces_psd():
    """The Welch spectrum of a generated series averages to the target PSD within 3 %."""
    dt = LOOP.dt
    x = noise.generate(N, dt, _psd, np.random.default_rng(0))
    f, s = noise.estimate_psd(x, dt, nperseg=2**12)
    band = (f > 1e5) & (f < 4e7)
    assert abs(np.mean(s[band] / _psd(f[band])) - 1) < 0.03


def test_beta_line_reproduces_lorentzian():
    """For white frequency noise and a long observation time the beta-line FWHM is pi * h0."""
    f = np.logspace(-1, 9, 200001)
    h0 = 3e3
    fwhm = noise.beta_line_linewidth(f, noise.white_and_flicker_psd(f, h0, 0.0), t_obs=10.0)
    assert np.isclose(fwhm, np.pi * h0, rtol=1e-2)


def test_default_loop_is_stable():
    """The default loop has a unity-gain frequency near 1 MHz and a phase margin above 45 degrees."""
    cav = thesis_cavity()
    f_u, margin = stability_margins(LOOP, cav.linewidth / 2)
    assert 0.8e6 < f_u < 1.2e6
    assert margin > 45


def _transfer(cav, x, discriminator):
    result = simulate(x, LOOP, cav.linewidth / 2, discriminator)
    f, h = noise.estimate_transfer(x[SKIP:], result["delta"][SKIP:], LOOP.dt, nperseg=2**14)
    f, h = f[1:], h[1:]  # drop f = 0, where the integrator makes L infinite
    predicted = 1 / (1 + open_loop_gain(f, LOOP, cav.linewidth / 2, discrete=True))
    return f, h / predicted


def test_linear_loop_matches_theory():
    """With a linear discriminator the simulated noise suppression equals 1 / (1 + L).

    The suppression is estimated from the cross-spectrum of the
    free-running and locked detunings. Between 100 kHz and 20 MHz it agrees
    with the analytic open-loop gain to 1 %; below 100 kHz the suppression
    exceeds 20 dB and the spectral estimate is less precise.
    """
    cav = thesis_cavity()
    x = noise.generate(N, LOOP.dt, _psd, np.random.default_rng(1))
    f, ratio = _transfer(cav, x, None)
    band = (f > 1e5) & (f < 2e7)
    assert np.max(np.abs(ratio[band] - 1)) < 0.01
    low = (f > 2e4) & (f <= 1e5)
    assert np.max(np.abs(ratio[low] - 1)) < 0.05


def test_pdh_discriminator_is_linear_for_small_noise():
    """With the full PDH curve and 5 % of the noise, the loop again follows linear theory.

    This confirms that the tabulated error curve has unit slope at
    resonance. At the full noise level the curvature of the error signal
    reduces the effective gain by a few per cent.
    """
    cav = thesis_cavity()
    x = 0.05 * noise.generate(N, LOOP.dt, _psd, np.random.default_rng(1))
    f, ratio = _transfer(cav, x, _disc(cav))
    band = (f > 1e5) & (f < 2e7)
    assert np.max(np.abs(ratio[band] - 1)) < 0.01


def test_lock_holds_on_resonance():
    """With the assumed laser noise the lock stays on the central slope and on resonance.

    The largest excursion of the cavity-filtered detuning must stay below
    the half-width (375 kHz), where the error signal turns over, and the
    mean must be within 1 kHz of zero.

    The noise reduction is tested below 100 kHz, where the loop gain is
    large, and must be at least a factor of five in rms. The total rms is
    not a useful measure here: most of the locked residual is white
    frequency noise above the ~1 MHz loop bandwidth, which no feedback can
    remove, while the free-running drift that the lock removes grows over
    seconds and is only partly represented in a 2.6 ms record.
    """
    from scipy import signal

    cav = thesis_cavity()
    disc = _disc(cav)
    x = noise.generate(N, LOOP.dt, _psd, np.random.default_rng(2))
    locked = simulate(x, LOOP, cav.linewidth / 2, disc)["delta_cav"][SKIP:]
    free = simulate(x, LOOP, cav.linewidth / 2, disc, engage=N)["delta_cav"][SKIP:]
    assert np.max(np.abs(locked)) < cav.linewidth / 2
    assert abs(np.mean(locked)) < 1e3

    sos = signal.butter(4, 100e3, fs=1 / LOOP.dt, output="sos")
    low_locked = signal.sosfiltfilt(sos, locked)
    low_free = signal.sosfiltfilt(sos, free)
    assert np.std(low_locked) < 0.2 * np.std(low_free)


def test_lock_acquires_from_offset():
    """Engaged 300 kHz from resonance, the loop pulls the laser onto resonance within 50 us."""
    cav = thesis_cavity()
    disc = _disc(cav)
    x = 300e3 + noise.generate(N // 4, LOOP.dt, _psd, np.random.default_rng(3))
    engage = 1000
    d = simulate(x, LOOP, cav.linewidth / 2, disc, engage=engage)["delta_cav"]
    after = d[engage + int(50e-6 / LOOP.dt):]
    assert abs(np.mean(after)) < 2e3
