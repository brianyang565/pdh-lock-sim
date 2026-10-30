"""Figure 3: the error signal from slow to fast modulation.

Each panel shows the analytic error signal for one ratio of modulation
frequency to cavity linewidth, with the demodulation phase chosen to
maximise the slope at resonance. The detuning axis is in units of the
linewidth, so the thesis cavity (Omega / linewidth = 27) appears in the
last panel. The slope at resonance is quoted in units of the
fast-modulation limit 4 J0 J1 / linewidth.

In Spyder, open this file and press F5. From a terminal, run
python scripts/fig3_modulation_regimes.py from the repository root.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import style  # noqa: E402
from pdh_sim import thesis_cavity  # noqa: E402
from pdh_sim.pdh import error_signal, optimal_phase, slope_at_resonance  # noqa: E402

BETA = 1.08
RATIOS = [0.3, 1.0, 3.0, 20e6 / 750e3]  # Omega / linewidth; the last is the thesis cavity


def main():
    style.apply()
    cav = thesis_cavity()
    limit = 4 * special.j0(BETA) * special.j1(BETA) / cav.linewidth

    fig, axes = plt.subplots(1, len(RATIOS), figsize=(11, 3.0), sharey=True,
                             constrained_layout=True)
    for i, (ax, ratio) in enumerate(zip(axes, RATIOS)):
        omega = ratio * cav.linewidth
        phase = optimal_phase(cav, omega, BETA)
        slope = slope_at_resonance(cav, omega, BETA, phase) / limit

        x_max = max(ratio, 1.0) * 1.4
        x = np.linspace(-x_max, x_max, 4001)
        eps = error_signal(cav, x * cav.linewidth, omega, BETA, phase)

        ax.plot(x, eps, color=style.INK)
        for s in (-1, 1):
            ax.axvline(s * ratio, color=style.MUTED, ls="--", lw=0.8)
        label = f"$\\Omega/\\delta\\nu$ = {ratio:.3g}"
        if i == len(RATIOS) - 1:
            label += " (thesis)"
        ax.set_title(f"({'abcd'[i]}) {label}", loc="left")
        ax.text(0.04, 0.96, f"slope = {slope:.2f}\nof fast limit",
                transform=ax.transAxes, va="top", color=style.TEXT)
        ax.set_xlabel(r"Detuning ($\delta\nu$)")
        print(f"Omega/linewidth = {ratio:6.3f}: optimal phase {phase:+.3f} rad, "
              f"slope {slope:.3f} of 4 J0 J1 / linewidth")
    axes[0].set_ylabel(r"Error signal ($P_0$)")

    out = ROOT / "figures" / "fig3_modulation_regimes.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
