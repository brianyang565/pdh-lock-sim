"""Figure 8: comparison with the measured noise of the cavity-transmitted light.

The thesis (Sec. 3) models the frequency noise of the cavity-transmitted
780 nm light as S = h0 exp(-A f), with h0 = 0.026(5) MHz and
A = 1.0(9) MHz^-1. The black line is this model at the central values; the
shaded band spans A = 0.1 to 1.9 MHz^-1 at fixed h0.

Two hypotheses for the origin of the measured level are fitted over
10 kHz - 3 MHz, each with the loop gain G as a second free parameter:

  H1  the free-running laser is noisier than assumed (noise scaled by k);
  H2  white noise S_e enters at the error-signal point (detector,
      electronics, residual amplitude modulation) with the assumed laser.

The measured level implies a residual detuning of about 160 kHz rms,
roughly 45 % of the cavity half-width, where the PDH error curve is
already noticeably curved. Linear theory therefore misestimates the
spectrum by up to a factor of 3 below 100 kHz, so each hypothesis is
fitted in two stages: a linear-theory fit supplies the starting point, and
a fit to the full nonlinear time-domain simulation refines it. The curves
shown are the simulated spectra for the final parameters, averaged in
logarithmic frequency bins. Below 10 kHz the two hypotheses separate:
laser noise is suppressed increasingly by the integrator, whereas
error-point noise is written onto the laser at all frequencies within the
loop bandwidth.

In Spyder, open this file and press F5 (about 90 s).
"""

import dataclasses
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import style  # noqa: E402
from pdh_sim import noise, thesis_cavity  # noqa: E402
from pdh_sim.config import (  # noqa: E402
    A_THESIS, A_THESIS_RANGE, BETA, H0, H0_THESIS, H1, LOOP, OMEGA_MOD)
from pdh_sim.loop import build_discriminator, locked_spectra, simulate, stability_margins  # noqa: E402
from pdh_sim.pdh import optimal_phase  # noqa: E402

A_RANGE = A_THESIS_RANGE
FIT_BAND = (1e4, 3e6)


def thesis_model(f, a=A_THESIS):
    return H0_THESIS * np.exp(-a * f)


def linear_fit(kind, cav):
    """Least-squares fit in log space of (log10 scale, log10 gain) using linear theory."""
    corner = cav.linewidth / 2
    f = np.logspace(np.log10(FIT_BAND[0]), np.log10(FIT_BAND[1]), 300)
    target = thesis_model(f)
    s_free = noise.white_and_flicker_psd(f, H0, H1)

    def model(v):
        p = dataclasses.replace(LOOP, gain=10 ** v[1])
        if kind == "laser":
            return locked_spectra(f, 10 ** v[0] * s_free, 0.0, p, corner)[1]
        return locked_spectra(f, s_free, 10 ** v[0], p, corner)[1]

    def cost(v):
        if not 0.05 < 10 ** v[1] < 6.0:
            return 1e9
        return np.sum(np.log(model(v) / target) ** 2)

    start = 1.5 if kind == "laser" else 4.3
    best = min((optimize.minimize(cost, [start, g0], method="Nelder-Mead",
                                  options=dict(xatol=1e-4, fatol=1e-10, maxiter=4000))
                for g0 in (-1.0, -0.5, 0.0, 0.3, 0.6)), key=lambda r: r.fun)
    return best.x


def simulated_spectrum(kind, v, x, unit_noise, disc, corner, nperseg):
    """Log-binned spectrum of the transmitted-light detuning for parameters v."""
    p = dataclasses.replace(LOOP, gain=10 ** v[1])
    if kind == "laser":
        d = simulate(np.sqrt(10 ** v[0]) * x, p, corner, disc)["delta_cav"]
    else:
        d = simulate(x, p, corner, disc, error_noise=np.sqrt(10 ** v[0]) * unit_noise)["delta_cav"]
    f, s = noise.estimate_psd(d[2**14:], LOOP.dt, nperseg=nperseg)
    return noise.log_bin(f, s, per_decade=12)


def nonlinear_fit(kind, v0, disc, corner):
    """Refine the fit with the full time-domain simulation (fixed noise realisation)."""
    n = 2**19
    x = noise.generate(n, LOOP.dt, lambda f: noise.white_and_flicker_psd(f, H0, H1),
                       np.random.default_rng(21))
    unit = noise.generate(n, LOOP.dt, np.ones_like, np.random.default_rng(22))

    def cost(v):
        if not 0.05 < 10 ** v[1] < 6.0:
            return 1e9
        fb, sb = simulated_spectrum(kind, v, x, unit, disc, corner, nperseg=2**13)
        band = (fb >= FIT_BAND[0]) & (fb <= FIT_BAND[1])
        return np.mean(np.log(sb[band] / thesis_model(fb[band])) ** 2)

    r = optimize.minimize(cost, v0, method="Nelder-Mead",
                          options=dict(xatol=0.01, fatol=1e-4, maxiter=60))
    return r.x, float(np.exp(np.sqrt(r.fun)))


def main():
    style.apply()
    cav = thesis_cavity()
    corner = cav.linewidth / 2
    dt = LOOP.dt
    disc = build_discriminator(cav, OMEGA_MOD, BETA, optimal_phase(cav, OMEGA_MOD, BETA))
    psd = lambda f: noise.white_and_flicker_psd(f, H0, H1)  # noqa: E731

    results = {}
    for kind in ("laser", "error"):
        v_lin = linear_fit(kind, cav)
        v, misfit = nonlinear_fit(kind, v_lin, disc, corner)
        results[kind] = (v_lin, v, misfit)
        print(f"{kind}: linear fit scale {10 ** v_lin[0]:.3g}, G {10 ** v_lin[1]:.2f} -> "
              f"simulation fit scale {10 ** v[0]:.3g}, G {10 ** v[1]:.2f}")

    # long runs with the final parameters, for the figure
    n = 2**21
    x = noise.generate(n, dt, psd, np.random.default_rng(31))
    unit = noise.generate(n, dt, np.ones_like, np.random.default_rng(32))
    curves = {kind: simulated_spectrum(kind, results[kind][1], x, unit, disc, corner, 2**17)
              for kind in results}
    f = np.logspace(2, 7, 600)
    k, g1 = 10 ** results["laser"][1][0], 10 ** results["laser"][1][1]
    s_e, g2 = 10 ** results["error"][1][0], 10 ** results["error"][1][1]
    r1, r2 = results["laser"][2], results["error"][2]

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.fill_between(f, thesis_model(f, A_RANGE[0]), thesis_model(f, A_RANGE[1]),
                    color="#e6e6e6", lw=0, label=r"Thesis model, $A$ = 0.1–1.9 MHz$^{-1}$")
    ax.loglog(f, thesis_model(f), color="black", lw=2.0, label="Thesis model, central values")
    for kind, colour, label in (
        ("laser", style.SERIES[0], f"H1: laser noise ×{k:.0f}, G = {g1:.1f}"),
        ("error", style.SERIES[1], f"H2: error-point noise {np.sqrt(s_e):.0f} Hz/√Hz, G = {g2:.1f}"),
    ):
        fb, sb = curves[kind]
        keep = (fb > 2e2) & (fb < 1e7)
        ax.loglog(fb[keep], sb[keep], color=colour, lw=2.2, label=label)
    ax.axvspan(*FIT_BAND, color="#eef3fa", zorder=0)
    ax.text(np.sqrt(FIT_BAND[0] * FIT_BAND[1]), 3e5, "fit range", ha="center", color=style.TEXT)
    ax.text(3e2, 2.5e2, "H1 and H2 separate\nbelow 10 kHz", color=style.TEXT)
    ax.set_xlim(2e2, 1e7)
    ax.set_ylim(1e1, 1e6)
    ax.set_xlabel("Fourier frequency (Hz)")
    ax.set_ylabel(r"Frequency-noise PSD of transmitted light (Hz$^2$/Hz)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)

    for name, g, rf in (("H1", g1, r1), ("H2", g2, r2)):
        f_u, pm = stability_margins(dataclasses.replace(LOOP, gain=g), corner)
        print(f"{name}: G = {g:.2f}, unity gain {f_u / 1e3:.0f} kHz, phase margin {pm:.0f} deg, "
              f"rms misfit factor {rf:.2f}")
    fg = np.logspace(0, 9, 400001)
    print(f"H1: free-running noise x{k:.1f}; implied free-running linewidth (1 ms) "
          f"{noise.beta_line_linewidth(fg, k * psd(fg), 1e-3) / 1e3:.0f} kHz")
    print(f"H2: error-point noise {s_e:.3g} Hz^2/Hz = {np.sqrt(s_e):.0f} Hz/rtHz")
    f1, s1 = curves["laser"]
    f2, s2 = curves["error"]
    for f0 in (1e3, 3e3, 1e4):
        a = s1[np.argmin(abs(f1 - f0))]
        b = s2[np.argmin(abs(f2 - f0))]
        print(f"  near {f0 / 1e3:4.0f} kHz: H1 {a:8.0f}, H2 {b:8.0f} Hz^2/Hz (ratio {b / a:.1f})")

    out = ROOT / "figures" / "fig8_thesis_comparison.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
