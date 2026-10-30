"""Figure 1: reflection response of the reference cavity.

(a) Reflected power |F|^2 near resonance, with the measured FWHM marked.
(b) Reflection phase arg F over the same range.
(c) Reflected power over +/-30 MHz, with the 20 MHz PDH sideband
    frequencies marked, which shows that the sidebands are reflected
    almost entirely when the carrier is on resonance.

In Spyder, open this file and press F5. From a terminal, run
python scripts/fig1_cavity_response.py from the repository root.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pdh_sim import measure_fwhm, thesis_cavity  # noqa: E402

INK = "#1f4e79"
MUTED = "#6b6b6b"
GRID = "#e3e3e3"
OMEGA_PDH = 20e6  # PDH modulation frequency, thesis Sec. 2.4.2

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": "#222222",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
    }
)


def main():
    cav = thesis_cavity()

    near = np.linspace(-3e6, 3e6, 6001)
    wide = np.linspace(-30e6, 30e6, 60001)
    F_near = cav.reflection(near)
    F_wide = cav.reflection(wide)

    dip = 1.0 - np.abs(F_near) ** 2
    fwhm = measure_fwhm(near, dip)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True)

    # (a) reflected power near resonance
    ax = axes[0]
    ax.plot(near / 1e6, np.abs(F_near) ** 2, color=INK)
    ax.annotate(
        "",
        xy=(fwhm / 2e6, 0.5),
        xytext=(-fwhm / 2e6, 0.5),
        arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.9),
    )
    ax.text(
        0.75, 0.47, f"FWHM = {fwhm / 1e3:.1f} kHz", ha="left", va="center", color="#222222"
    )
    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel(r"Reflected power $|F|^2$")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("(a)", loc="left")

    # (b) reflection phase
    ax = axes[1]
    ax.plot(near / 1e6, np.angle(F_near) / np.pi, color=INK)
    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel(r"Reflection phase $\arg F$ ($\pi$ rad)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("(b)", loc="left")

    # (c) wide view with PDH sideband frequencies
    ax = axes[2]
    ax.plot(wide / 1e6, np.abs(F_wide) ** 2, color=INK)
    for s in (-1, 1):
        ax.axvline(s * OMEGA_PDH / 1e6, color=MUTED, ls="--", lw=0.9)
    ax.text(OMEGA_PDH / 1e6, 0.15, r" $+\Omega$", color=MUTED, ha="left")
    ax.text(-OMEGA_PDH / 1e6, 0.15, r"$-\Omega$ ", color=MUTED, ha="right")
    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel(r"Reflected power $|F|^2$")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("(c)", loc="left")

    out = ROOT / "figures" / "fig1_cavity_response.png"
    fig.savefig(out, dpi=200)
    print(f"measured FWHM = {fwhm:.1f} Hz; expected FSR/F = {cav.linewidth:.1f} Hz")
    print(f"saved {out}")
    plt.show()  # displays the figure in Spyder's Plots pane


if __name__ == "__main__":
    main()
