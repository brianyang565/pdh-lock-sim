"""Checks of the PDH error signal.

The first group checks the time-domain cavity simulation against the
steady-state cavity model. The second group checks the analytic error
signal against the time-domain simulation and against the textbook
fast-modulation limit. The last group checks that the differences seen in
a swept scan are the expected dynamical lags, not errors.

Assumed modulation depth: beta = 1.08 rad, which maximises J0 J1 and hence
the error-signal slope (Black 2001). The modulation depth of the
experiment is not recorded in the thesis.
"""

import numpy as np
from scipy import special

from pdh_sim import thesis_cavity
from pdh_sim.pdh import (
    demodulate,
    error_signal,
    lowpass_delay,
    optimal_phase,
    phase_modulated_laser,
    simulate_reflection,
    slope_at_resonance,
    static_error_time_domain,
)

OMEGA = 20e6  # PDH modulation frequency, thesis Sec. 2.4.2
BETA = 1.08   # assumed modulation depth


def test_time_domain_cavity_matches_steady_state():
    """For a monochromatic laser the round-trip simulation reproduces F(delta).

    After 20 000 round trips (13 us, about 30 field time constants) the
    ratio of reflected to incident field must equal the steady-state
    reflection coefficient of cavity.py.
    """
    cav = thesis_cavity()
    t = np.arange(20000) * cav.round_trip_time
    for delta in (0.0, 3e5, -1.1e6, 7e6):
        e_ref = simulate_reflection(cav, 2 * np.pi * delta * t)
        ratio = e_ref[-1] / np.exp(-2j * np.pi * delta * t[-1])
        assert abs(ratio - cav.reflection(delta)) < 1e-10


def test_error_signal_time_domain_matches_analytic():
    """The demodulated photodiode signal equals Re[c1 exp(i phase)] at fixed detunings.

    The two calculations share no code beyond cavity.py: one sums Bessel
    sidebands in the frequency domain, the other steps the intracavity
    field in time and mixes the photodiode power with a local oscillator.
    Agreement to 1e-9 therefore checks both.
    """
    cav = thesis_cavity()
    for phase in (optimal_phase(cav, OMEGA, BETA), 0.4):
        for delta in (0.0, 1e5, -4e5, 2e6, 19.5e6, -21e6):
            analytic = error_signal(cav, delta, OMEGA, BETA, phase)
            simulated = static_error_time_domain(cav, delta, OMEGA, BETA, phase)
            assert abs(analytic - simulated) < 1e-9


def test_error_signal_is_zero_on_resonance_and_odd():
    """eps(-delta) = -eps(delta) for any demodulation phase, so eps(0) = 0.

    This follows from F(-delta) = F(delta)^*, which makes c1 an odd
    function of the detuning. The lock point therefore coincides with the
    cavity resonance regardless of the demodulation phase.
    """
    cav = thesis_cavity()
    delta = np.linspace(-30e6, 30e6, 2001)
    for phase in (0.0, 0.7, optimal_phase(cav, OMEGA, BETA)):
        eps = error_signal(cav, delta, OMEGA, BETA, phase)
        assert np.allclose(eps, -eps[::-1], atol=1e-14)


def test_slope_matches_fast_modulation_limit():
    """At the optimal phase the slope approaches 4 J0(beta) J1(beta) / linewidth.

    In the fast-modulation limit the sidebands are reflected with F = -1 and
    only the carrier phase varies, which gives this slope per unit P0. With
    Omega / linewidth = 27 the correction is below 0.1 %.
    """
    cav = thesis_cavity()
    phase = optimal_phase(cav, OMEGA, BETA)
    slope = slope_at_resonance(cav, OMEGA, BETA, phase)
    expected = 4 * special.j0(BETA) * special.j1(BETA) / cav.linewidth
    assert np.isclose(slope, expected, rtol=1e-3)


def test_optimal_phase_offset_from_quadrature():
    """The optimal phase differs from -pi/2 by approximately linewidth / (2 Omega).

    With a Lorentzian approximation to F near resonance, the sidebands are
    reflected with F(+-Omega) = -1 +- i linewidth / (2 Omega). This small
    imaginary part rotates the error-signal slope away from pure
    quadrature by linewidth / (2 Omega) = 0.019 rad for the thesis cavity.
    """
    cav = thesis_cavity()
    phase = optimal_phase(cav, OMEGA, BETA)
    expected = -np.pi / 2 + cav.linewidth / (2 * OMEGA)
    assert abs(phase - expected) < 1e-3


def _swept_deviation(cav, rate, cutoff=1e6):
    """Maximum |simulated - analytic| within +-2 MHz for a linear sweep, per unit peak.

    Returns the deviation from the static curve and from the static curve
    shifted by the combined lag of the low-pass filter and the cavity field.
    """
    phase = optimal_phase(cav, OMEGA, BETA)
    start, stop = -4e6, 4e6
    n = int((stop - start) / rate * cav.fsr)
    t = np.arange(n) * cav.round_trip_time
    delta = start + rate * t
    power = np.abs(simulate_reflection(cav, phase_modulated_laser(t, delta, OMEGA, BETA))) ** 2
    eps = demodulate(t, power, OMEGA, phase, cutoff)

    lag = lowpass_delay(cutoff) + cav.field_time_constant
    static = error_signal(cav, delta, OMEGA, BETA, phase)
    shifted = error_signal(cav, delta - rate * lag, OMEGA, BETA, phase)
    window = np.abs(delta) < 2e6
    peak = np.max(np.abs(static))
    return (np.max(np.abs(eps - static)[window]) / peak,
            np.max(np.abs(eps - shifted)[window]) / peak)


def test_swept_scan_deviation_is_a_dynamical_lag():
    """In a swept scan the deviation from the static curve is a lag set by the sweep rate.

    Two properties identify it. First, halving the sweep rate halves the
    deviation, so it is first order in the rate. Second, shifting the
    static curve by the filter delay plus the cavity field time constant
    removes most of it. The remainder arises because the cavity delay
    depends on the detuning and is largest on resonance.
    """
    cav = thesis_cavity()
    raw_fast, corrected_fast = _swept_deviation(cav, rate=50e9)   # 50 MHz/ms
    raw_slow, corrected_slow = _swept_deviation(cav, rate=25e9)   # 25 MHz/ms
    assert np.isclose(raw_fast / raw_slow, 2.0, rtol=0.1)
    assert corrected_slow < 0.3 * raw_slow
    assert corrected_slow < 0.03
