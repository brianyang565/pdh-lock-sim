"""Steady-state optical response of a lossless, symmetric Fabry-Perot cavity.

All frequencies are in Hz. The detuning ``delta`` is the laser frequency
measured from the nearest cavity resonance, so ``delta = 0`` is on resonance.

The reference-cavity parameters are those of the experiment described in
Sec. 2.4.2 of the thesis: free spectral range 1.5 GHz and finesse 2000.

References
----------
E. D. Black, "An introduction to Pound-Drever-Hall laser frequency
stabilization," Am. J. Phys. 69, 79 (2001).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cavity:
    """A lossless two-mirror cavity with identical mirrors.

    Parameters
    ----------
    fsr : float
        Free spectral range in Hz.
    finesse : float
        Coefficient finesse, defined by F = pi * sqrt(R) / (1 - R).
    """

    fsr: float
    finesse: float

    # ------------------------------------------------------------------
    # Derived mirror and cavity quantities
    # ------------------------------------------------------------------
    @property
    def intensity_reflectivity(self) -> float:
        """Mirror intensity reflectivity R, obtained by inverting the finesse.

        Setting x = sqrt(R), the definition F = pi x / (1 - x^2) becomes the
        quadratic F x^2 + pi x - F = 0, whose positive root is
        x = (-pi + sqrt(pi^2 + 4 F^2)) / (2 F).
        """
        f = self.finesse
        x = (-np.pi + np.sqrt(np.pi**2 + 4.0 * f**2)) / (2.0 * f)
        return float(x**2)

    @property
    def r(self) -> float:
        """Mirror amplitude reflectivity r = sqrt(R)."""
        return float(np.sqrt(self.intensity_reflectivity))

    @property
    def linewidth(self) -> float:
        """Resonance full width at half maximum, FSR / finesse, in Hz.

        This is the high-finesse approximation. The exact Airy width,
        (FSR / pi) * 2 * arcsin((1 - R) / (2 sqrt(R))), differs from it by
        a relative amount of order 1 / F^2, which is about 1e-7 here.
        """
        return self.fsr / self.finesse

    @property
    def photon_lifetime(self) -> float:
        """Intracavity field-energy decay time 1 / (2 pi * linewidth), in s."""
        return 1.0 / (2.0 * np.pi * self.linewidth)

    @property
    def field_time_constant(self) -> float:
        """Decay time of the intracavity field amplitude, 1 / (pi * linewidth), in s.

        The field decays at half the rate of the stored energy, so this is
        twice the photon lifetime. It sets the delay with which the
        reflected field follows a change of the laser frequency, and it
        corresponds to a single-pole low-pass at the half-width,
        linewidth / 2.
        """
        return 1.0 / (np.pi * self.linewidth)

    @property
    def round_trip_time(self) -> float:
        """Round-trip time 1 / FSR, in s."""
        return 1.0 / self.fsr

    # ------------------------------------------------------------------
    # Field response
    # ------------------------------------------------------------------
    def reflection(self, delta):
        """Complex amplitude reflection coefficient F(delta).

        For a lossless cavity with identical mirrors, the reflected field
        divided by the incident field is

            F = r (exp(i phi) - 1) / (1 - r^2 exp(i phi)),

        where phi = 2 pi delta / FSR is the round-trip phase measured from
        resonance.
        """
        phi = 2.0 * np.pi * np.asarray(delta, dtype=float) / self.fsr
        z = np.exp(1j * phi)
        r = self.r
        return r * (z - 1.0) / (1.0 - r**2 * z)

    def transmission(self, delta):
        """Complex amplitude transmission coefficient T(delta).

        With mirror transmission t^2 = 1 - r^2 and the phase reference taken
        at the output mirror, T = t^2 exp(i phi / 2) / (1 - r^2 exp(i phi)).
        For a lossless cavity |F|^2 + |T|^2 = 1 at every detuning.
        """
        phi = 2.0 * np.pi * np.asarray(delta, dtype=float) / self.fsr
        r2 = self.intensity_reflectivity
        return (1.0 - r2) * np.exp(0.5j * phi) / (1.0 - r2 * np.exp(1j * phi))


def thesis_cavity() -> Cavity:
    """Reference cavity of the experiment (thesis Sec. 2.4.2)."""
    return Cavity(fsr=1.5e9, finesse=2000.0)


def measure_fwhm(delta, power) -> float:
    """Full width at half maximum of a single peak sampled on a grid.

    The two half-maximum crossings are located by linear interpolation
    between neighbouring samples, so the result does not assume any line
    shape. ``power`` must contain exactly one peak, and the grid must
    extend below half maximum on both sides of it.
    """
    delta = np.asarray(delta, dtype=float)
    power = np.asarray(power, dtype=float)
    i_peak = int(np.argmax(power))
    half = 0.5 * power[i_peak]

    # Left crossing: last sample below half maximum before the peak.
    left = np.nonzero(power[:i_peak] < half)[0]
    right = np.nonzero(power[i_peak:] < half)[0]
    if left.size == 0 or right.size == 0:
        raise ValueError("grid does not extend below half maximum on both sides")
    i0 = left[-1]
    i1 = i_peak + right[0]

    def crossing(ia, ib):
        return np.interp(half, [power[ia], power[ib]], [delta[ia], delta[ib]])

    x_left = crossing(i0, i0 + 1)
    x_right = np.interp(half, [power[i1], power[i1 - 1]], [delta[i1], delta[i1 - 1]])
    return float(x_right - x_left)
