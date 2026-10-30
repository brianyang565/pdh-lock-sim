"""Figure 6: frequency-noise spectra of the free-running and locked laser.

Thin coloured lines: Welch spectra from the time-domain simulation (full
PDH error curve, assumed laser noise, default loop).
Dashed black lines: linear theory from ``loop.locked_spectra``.
Dotted grey line: the beta-separation line 8 ln2 f / pi^2. Noise above it
broadens the laser line; noise below it contributes only to the wings.

In Spyder, open this file and press F5 (about 5 s).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import style  # noqa: E402
from pdh_sim import noise, thesis_cavity  # noqa: E402
from pdh_sim.config import BETA, H0, H1, LOOP, OMEGA_MOD  # noqa: E402
from pdh_sim.loop import build_discriminator, locked_spectra, simulate  # noqa: E402
from pdh_sim.pdh import optimal_phase  # noqa: E402

N = 2**21     # 21 ms
SKIP = 2**15
T_OBS = 1e-3  # observation time for the linewidths


def main():
    style.apply()
    cav = thesis_cavity()
    corner = cav.linewidth / 2
    dt = LOOP.dt
    disc = build_discriminator(cav, OMEGA_MOD, BETA, optimal_phase(cav, OMEGA_MOD, BETA))
    psd = lambda f: noise.white_and_flicker_psd(f, H0, H1)  # noqa: E731

    x = noise.generate(N, dt, psd, np.random.default_rng(11))
    res = simulate(x, LOOP, corner, disc)

    f_sim, s_free = noise.estimate_psd(x[SKIP:], dt, nperseg=2**17)
    _, s_delta = noise.estimate_psd(res["delta"][SKIP:], dt, nperseg=2**17)
    _, s_cav = noise.estimate_psd(res["delta_cav"][SKIP:], dt, nperseg=2**17)

    f = np.logspace(2, 8, 2000)
    th_free = psd(f)
    th_delta, th_cav = locked_spectra(f, th_free, 0.0, LOOP, corner)
    beta_line = 8 * np.log(2) * f / np.pi**2

    fg = np.logspace(0, 9, 400001)
    lw_free = noise.beta_line_linewidth(fg, psd(fg), T_OBS)
    sd, sc = locked_spectra(fg, psd(fg), 0.0, LOOP, corner)
    lw_locked = noise.beta_line_linewidth(fg, sd, T_OBS)

    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    band = (f_sim > 1e3) & (f_sim < 4e7)
    curves = [
        (s_free, th_free, "Free-running laser"),
        (s_delta, th_delta, "Locked laser"),
        (s_cav, th_cav, "Locked, transmitted through cavity"),
    ]
    for (s, th, label), colour in zip(curves, style.SERIES):
        ax.loglog(f_sim[band], s[band], color=colour, lw=0.9, label=label)
        ax.loglog(f, th, color="black", lw=0.9, ls="--")
    ax.loglog(f, beta_line, color=style.MUTED, lw=1.0, ls=":")
    ax.plot([], [], color="black", lw=0.9, ls="--", label="Linear theory")
    ax.plot([], [], color=style.MUTED, lw=1.0, ls=":", label=r"$\beta$-separation line")

    # direct labels
    ax.text(1.5e3, 3.5e5, "free-running", color=style.TEXT)
    ax.text(1.5e3, 3, "locked laser = transmitted\n(the curves coincide below 300 kHz)", color=style.TEXT)
    ax.text(6e6, 1.2e2, "transmitted", color=style.TEXT)
    ax.text(6e6, 1e4, "locked laser", color=style.TEXT)

    ax.set_xlim(1e3, 4e7)
    ax.set_ylim(1e-1, 1e7)
    ax.set_xlabel("Fourier frequency (Hz)")
    ax.set_ylabel(r"Frequency-noise PSD $S_\nu$ (Hz$^2$/Hz)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.text(
        0.02, 0.03,
        f"Linewidth ({T_OBS * 1e3:.0f} ms, $\\beta$-line): free-running {lw_free / 1e3:.0f} kHz; "
        f"locked, relative to cavity: {'< 1 kHz' if lw_locked < 1e3 else f'{lw_locked / 1e3:.1f} kHz'}",
        transform=ax.transAxes, color=style.TEXT,
    )

    print(f"free-running linewidth {lw_free / 1e3:.1f} kHz; locked {lw_locked:.1f} Hz "
          "(0 means the spectrum lies entirely below the beta line)")
    out = ROOT / "figures" / "fig6_spectra.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
