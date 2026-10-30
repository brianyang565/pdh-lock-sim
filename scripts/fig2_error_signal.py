"""Figure 2: PDH error signal, analytic calculation and simulated scan.

The simulated trace is what an oscilloscope would show while the laser
frequency is swept linearly through the resonance at 25 MHz/ms. The
photodiode power is computed from the round-trip cavity simulation, mixed
with the local oscillator and passed through a 4th-order, 1 MHz Butterworth
low-pass filter. The analytic curve is the steady-state error signal.

(a) Full scan over +-25 MHz, showing the central feature and the two
    sideband features at +-Omega.
(b) The central slope, where the simulated trace lags the analytic curve.
    The lag is the filter delay plus the cavity field time constant.

The simulation uses 3 million time steps and takes about 15 s.

In Spyder, open this file and press F5. From a terminal, run
python scripts/fig2_error_signal.py from the repository root.
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
from pdh_sim.pdh import (  # noqa: E402
    demodulate,
    error_signal,
    lowpass_delay,
    optimal_phase,
    phase_modulated_laser,
    simulate_reflection,
)

OMEGA = 20e6       # PDH modulation frequency, thesis Sec. 2.4.2
BETA = 1.08        # assumed modulation depth (maximises J0 J1)
RATE = 25e9        # sweep rate, Hz/s (25 MHz/ms)
SPAN = 25e6        # sweep from -SPAN to +SPAN
CUTOFF = 1e6       # low-pass cutoff after the mixer


def main():
    style.apply()
    cav = thesis_cavity()
    phase = optimal_phase(cav, OMEGA, BETA)

    # --- simulated scan -------------------------------------------------
    n = int(2 * SPAN / RATE * cav.fsr)
    t = np.arange(n) * cav.round_trip_time
    delta = -SPAN + RATE * t
    power = np.abs(simulate_reflection(cav, phase_modulated_laser(t, delta, OMEGA, BETA))) ** 2
    eps_sim = demodulate(t, power, OMEGA, phase, CUTOFF)
    del power
    # Discard the first 20 us, during which the cavity fills and the filter
    # starts up, and keep every 50th point (60 000 are enough to draw the trace).
    start = int(20e-6 * cav.fsr)
    d_sim, e_sim = delta[start::50], eps_sim[start::50]

    # --- analytic curve ---------------------------------------------------
    d_an = np.linspace(-SPAN, SPAN, 20001)
    e_an = error_signal(cav, d_an, OMEGA, BETA, phase)

    lag_t = lowpass_delay(CUTOFF) + cav.field_time_constant
    lag_f = RATE * lag_t

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    ax.plot(d_an / 1e6, e_an, color=style.ACCENT, lw=2.4, label="Analytic (steady state)")
    ax.plot(d_sim / 1e6, e_sim, color=style.INK, lw=1.0, label="Simulated scan")
    for s in (-1, 1):
        ax.axvline(s * OMEGA / 1e6, color=style.MUTED, ls="--", lw=0.8)
    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel(r"Error signal ($P_0$)")
    ax.set_title("(a)", loc="left")
    ax.legend(loc="upper left")

    ax = axes[1]
    zoom = np.abs(d_an) < 1.5e6
    zs = np.abs(d_sim) < 1.5e6
    ax.plot(d_an[zoom] / 1e3, e_an[zoom], color=style.ACCENT, lw=2.4, label="Analytic")
    ax.plot(d_sim[zs] / 1e3, e_sim[zs], color=style.INK, lw=1.2, label="Simulated scan")
    ax.set_xlabel("Detuning (kHz)")
    ax.set_title("(b)", loc="left")
    ax.text(
        0.03, 0.97,
        f"sweep {RATE / 1e9:.0f} MHz/ms\n"
        f"lag {lag_t * 1e9:.0f} ns = {lag_f / 1e3:.0f} kHz",
        transform=ax.transAxes, va="top", color=style.TEXT,
    )
    ax.legend(loc="lower right")

    out = ROOT / "figures" / "fig2_error_signal.png"
    fig.savefig(out, dpi=200)
    print(f"optimal demodulation phase = {phase:.4f} rad")
    print(f"lag: filter {lowpass_delay(CUTOFF) * 1e9:.0f} ns + cavity "
          f"{cav.field_time_constant * 1e9:.0f} ns = {lag_t * 1e9:.0f} ns ({lag_f / 1e3:.1f} kHz)")
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
