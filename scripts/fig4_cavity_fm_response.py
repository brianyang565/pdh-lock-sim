"""Figure 4: response of the error signal to laser frequency modulation.

Markers: the full round-trip optical simulation, with the laser frequency
modulated sinusoidally at f and the error signal demodulated at f.
Line: a single pole at the cavity half-width, linewidth / 2 = 375 kHz.

The agreement justifies modelling the cavity inside the feedback loop as
a low-pass filter followed by the static error curve.

In Spyder, open this file and press F5.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import style  # noqa: E402
from pdh_sim import thesis_cavity  # noqa: E402
from pdh_sim.config import BETA, OMEGA_MOD  # noqa: E402
from pdh_sim.pdh import frequency_modulation_response, optimal_phase  # noqa: E402

DIVISORS = [400, 200, 100, 80, 50, 40, 25, 20, 10, 8, 5]  # f = OMEGA_MOD / k


def main():
    style.apply()
    cav = thesis_cavity()
    phase = optimal_phase(cav, OMEGA_MOD, BETA)
    corner = cav.linewidth / 2

    f_pts = np.array([OMEGA_MOD / k for k in DIVISORS])
    h_pts = np.array([frequency_modulation_response(cav, f, OMEGA_MOD, BETA, phase) for f in f_pts])
    f_line = np.logspace(4, 7, 400)
    h_line = 1 / (1 + 1j * f_line / corner)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)

    ax = axes[0]
    ax.loglog(f_line, np.abs(h_line), color=style.ACCENT, label="Single pole at 375 kHz")
    ax.loglog(f_pts, np.abs(h_pts), "o", ms=6, mfc="white", mec=style.INK, mew=1.5,
              label="Optical simulation")
    ax.axvline(corner, color=style.MUTED, ls="--", lw=0.8)
    ax.set_xlabel("Modulation frequency (Hz)")
    ax.set_ylabel("Response (relative to static)")
    ax.set_title("(a)", loc="left")
    ax.legend(loc="lower left")

    ax = axes[1]
    ax.semilogx(f_line, np.degrees(np.angle(h_line)), color=style.ACCENT)
    ax.semilogx(f_pts, np.degrees(np.angle(h_pts)), "o", ms=6, mfc="white", mec=style.INK, mew=1.5)
    ax.axvline(corner, color=style.MUTED, ls="--", lw=0.8)
    ax.set_xlabel("Modulation frequency (Hz)")
    ax.set_ylabel("Phase (deg)")
    ax.set_ylim(-95, 5)
    ax.set_title("(b)", loc="left")

    for f, h in zip(f_pts, h_pts):
        rel = abs(h * (1 + 1j * f / corner) - 1)
        print(f"f = {f / 1e3:7.1f} kHz: |response| = {abs(h):.4f}, deviation from pole = {rel:.2e}")

    out = ROOT / "figures" / "fig4_cavity_fm_response.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
