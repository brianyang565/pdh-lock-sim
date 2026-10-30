"""Laser frequency noise: generation from a spectral model and spectral estimation.

Frequency noise is described by its one-sided power spectral density
S_nu(f) in Hz^2/Hz. The model used here has two terms,

    S_nu(f) = h0 + h1 / f,

where h0 is white frequency noise and h1 / f is flicker frequency noise.
White frequency noise alone produces a Lorentzian line of FWHM pi * h0,
which is the "intrinsic" linewidth; the flicker term adds the technical
noise that dominates the linewidth of external-cavity diode lasers on
longer timescales.

Values for the MOGLabs cat-eye laser are not specified by the
manufacturer beyond a typical linewidth below 100 kHz, so the parameters
used in the scripts are stated there as assumptions.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, signal


def white_and_flicker_psd(f, h0: float, h1: float):
    """One-sided frequency-noise PSD h0 + h1 / f, in Hz^2/Hz."""
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore"):
        return h0 + np.where(f > 0, h1 / f, 0.0)


def lorentzian_fwhm_to_h0(fwhm: float) -> float:
    """White frequency-noise level that produces a Lorentzian of the given FWHM."""
    return fwhm / np.pi


def generate(n: int, dt: float, psd, rng=None):
    """Gaussian frequency-noise time series with a prescribed one-sided PSD.

    Each Fourier component is drawn with independent Gaussian real and
    imaginary parts whose variance is set by psd(f_k); the inverse FFT then
    gives a real, stationary series. With sampling rate fs = 1/dt and
    frequency spacing df = fs/n, the component at f_k must satisfy
    E|X_k|^2 = S(f_k) fs n / 2, so that the series variance equals the
    integral of S over 0 < f < fs/2. The DC and Nyquist components are set
    to zero.

    Parameters
    ----------
    n : int
        Number of samples.
    dt : float
        Sampling interval in s.
    psd : callable
        Function returning S_nu(f) in Hz^2/Hz for an array of frequencies.
    rng : numpy.random.Generator, optional
        Random-number generator, for reproducible series.

    Returns
    -------
    nu : ndarray
        Frequency deviation in Hz, with zero mean.
    """
    rng = np.random.default_rng() if rng is None else rng
    fs = 1.0 / dt
    f = np.fft.rfftfreq(n, dt)
    amp = np.sqrt(np.asarray(psd(f), dtype=float) * fs * n / 2.0)
    amp[0] = 0.0
    if n % 2 == 0:
        amp[-1] = 0.0
    spectrum = amp * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)) / np.sqrt(2)
    return np.fft.irfft(spectrum, n)


def estimate_psd(x, dt: float, nperseg: int = 2**14):
    """One-sided PSD of a time series by Welch's method (Hann window, 50 % overlap)."""
    return signal.welch(x, fs=1.0 / dt, nperseg=nperseg, scaling="density")


def estimate_transfer(x, y, dt: float, nperseg: int = 2**14):
    """Transfer function from x to y, H = S_xy / S_xx, estimated by Welch's method.

    For a noiseless linear system this estimate is unbiased, and because the
    same realisation of x drives y, its scatter is small even for short
    records.
    """
    fs = 1.0 / dt
    f, sxx = signal.welch(x, fs=fs, nperseg=nperseg)
    _, sxy = signal.csd(x, y, fs=fs, nperseg=nperseg)
    return f, sxy / sxx


def beta_line_linewidth(f, psd_values, t_obs: float) -> float:
    """Laser linewidth (FWHM) from a frequency-noise PSD by the beta-separation line.

    Di Domenico, Schilt and Thomann [Appl. Opt. 49, 4801 (2010)] showed that
    only the part of S_nu(f) lying above the line 8 ln2 f / pi^2 broadens
    the line shape; the part below it contributes to the wings only. The
    FWHM is then approximately sqrt(8 ln2 A), where A is the area of S_nu
    above that line, integrated from 1 / t_obs. For white frequency noise
    and t_obs -> infinity this reproduces the Lorentzian width pi * h0
    exactly; a finite observation time excludes the lowest frequencies and
    gives a slightly smaller value.

    Parameters
    ----------
    f, psd_values : arrays
        Frequency grid (Hz, ascending) and one-sided S_nu(f) in Hz^2/Hz.
    t_obs : float
        Observation time in s, which sets the lower integration limit.
    """
    f = np.asarray(f, dtype=float)
    s = np.asarray(psd_values, dtype=float)
    mask = (f >= 1.0 / t_obs) & (s > 8 * np.log(2) * f / np.pi**2)
    area = integrate.trapezoid(np.where(mask, s, 0.0), f)
    return float(np.sqrt(8 * np.log(2) * area))


def log_bin(f, s, per_decade: int = 20):
    """Average a spectrum in logarithmically spaced frequency bins.

    Welch spectra have many more points per decade at high frequency than
    at low frequency; averaging in bins of equal logarithmic width gives a
    smooth curve and weights every decade equally in a fit. Empty bins are
    dropped. Returns the mean frequency and mean PSD of each bin.
    """
    f = np.asarray(f, dtype=float)
    s = np.asarray(s, dtype=float)
    pos = f > 0
    f, s = f[pos], s[pos]
    edges = np.logspace(np.log10(f[0]), np.log10(f[-1]) + 1e-9,
                        int(np.ceil(per_decade * np.log10(f[-1] / f[0]))) + 1)
    idx = np.digitize(f, edges)
    keep = [i for i in range(1, edges.size) if np.any(idx == i)]
    return (np.array([f[idx == i].mean() for i in keep]),
            np.array([s[idx == i].mean() for i in keep]))
