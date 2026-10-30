"""Checks of the cavity model against analytic properties.

Each test states the property it checks, so that a failure identifies
which part of the model is wrong.
"""

import numpy as np

from pdh_sim.cavity import Cavity, measure_fwhm, thesis_cavity


def test_reflectivity_reproduces_finesse():
    """Inverting the finesse and re-evaluating pi sqrt(R) / (1 - R) returns F."""
    cav = thesis_cavity()
    R = cav.intensity_reflectivity
    assert np.isclose(np.pi * np.sqrt(R) / (1.0 - R), cav.finesse, rtol=1e-12)


def test_linewidth_from_transmission_is_750_khz():
    """The FWHM measured from |T|^2 equals FSR / F = 750 kHz (thesis Sec. 2.4.2).

    The grid spacing is 1 kHz, about 1/750 of the linewidth, so the linear
    interpolation error is far below the 1e-4 tolerance.
    """
    cav = thesis_cavity()
    delta = np.linspace(-3e6, 3e6, 6001)
    fwhm = measure_fwhm(delta, np.abs(cav.transmission(delta)) ** 2)
    assert np.isclose(fwhm, 750e3, rtol=1e-4)


def test_linewidth_from_reflection_dip_matches():
    """The reflection dip, 1 - |F|^2, has the same width as the transmission peak."""
    cav = thesis_cavity()
    delta = np.linspace(-3e6, 3e6, 6001)
    fwhm = measure_fwhm(delta, 1.0 - np.abs(cav.reflection(delta)) ** 2)
    assert np.isclose(fwhm, cav.linewidth, rtol=1e-4)


def test_energy_conservation():
    """A lossless cavity reflects or transmits all incident power."""
    cav = thesis_cavity()
    delta = np.linspace(-0.6 * cav.fsr, 0.6 * cav.fsr, 20001)
    total = np.abs(cav.reflection(delta)) ** 2 + np.abs(cav.transmission(delta)) ** 2
    assert np.allclose(total, 1.0, atol=1e-12)


def test_zero_reflection_on_resonance():
    """For identical lossless mirrors the cavity is impedance matched: F(0) = 0."""
    assert abs(thesis_cavity().reflection(0.0)) < 1e-15


def test_response_is_periodic_in_fsr():
    """Shifting the detuning by one free spectral range leaves F unchanged."""
    cav = thesis_cavity()
    delta = np.linspace(-2e6, 2e6, 401)
    assert np.allclose(cav.reflection(delta), cav.reflection(delta + cav.fsr), atol=1e-9)


def test_reflection_phase_winds_through_2pi():
    """Across an impedance-matched resonance the reflection phase changes by 2 pi.

    This phase excursion, which is confined to roughly one linewidth, is
    what the PDH technique converts into an error signal.
    """
    cav = thesis_cavity()
    delta = np.linspace(-50 * cav.linewidth, 50 * cav.linewidth, 20001)
    phase = np.unwrap(np.angle(cav.reflection(delta)))
    assert np.isclose(abs(phase[-1] - phase[0]), 2 * np.pi, atol=0.05)


def test_sidebands_are_far_off_resonance():
    """At the 20 MHz PDH modulation frequency the sidebands are almost fully reflected.

    Omega / linewidth = 20 MHz / 750 kHz is about 27, which places the
    experiment in the fast-modulation regime.
    """
    cav = thesis_cavity()
    assert np.abs(cav.reflection(20e6)) ** 2 > 0.99


def test_generic_cavity_linewidth():
    """The linewidth check is not specific to one parameter set."""
    cav = Cavity(fsr=10e9, finesse=300.0)
    delta = np.linspace(-150e6, 150e6, 30001)
    fwhm = measure_fwhm(delta, np.abs(cav.transmission(delta)) ** 2)
    # At F = 300 the exact Airy width differs from FSR / F by about 1e-5.
    R = cav.intensity_reflectivity
    exact = cav.fsr / np.pi * 2 * np.arcsin((1 - R) / (2 * np.sqrt(R)))
    assert np.isclose(fwhm, exact, rtol=1e-4)
