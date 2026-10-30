"""Figure 5: the closed loop.

(a) Open-loop gain |L(f)| of the default loop, with the unity-gain
    frequency and the phase margin.
(b) Noise suppression |delta / nu_free|^2: analytic 1 / |1 + L|^2 and the
    value measured in the time-domain simulation with the full PDH error
    curve and the assumed laser noise. Values below 1 (0 dB) mean the loop
    reduces the noise; the peak above 1 near the unity-gain frequency is the
    servo bump, where feedback amplifies the noise.
(c) Laser detuning from resonance as seen through the cavity, averaged
    over 1 us. The laser starts 300 kHz from resonance, and the loop is
    switched on at t = 0.5 ms.

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
from pdh_sim import noise, thesis_cavity  # noqa: E402
from pdh_sim.config import BETA, H0, H1, LOOP, OMEGA_MOD  # noqa: E402
from pdh_sim.loop import build_discriminator, open_loop_gain, simulate, stability_margins  # noqa: E402
from pdh_sim.pdh import optimal_phase  # noqa: E402

N = 2**19          # 5.2 ms
SKIP = 2**14
OFFSET = 300e3     # initial detuning for panel (c)
ENGAGE_T = 0.5e-3  # s


def main():
    style.apply()
    cav = thesis_cavity()
    corner = cav.linewidth / 2
    disc = build_discriminator(cav, OMEGA_MOD, BETA, optimal_phase(cav, OMEGA_MOD, BETA))
    psd = lambda f: noise.white_and_flicker_psd(f, H0, H1)  # noqa: E731
    dt = LOOP.dt

    f_u, margin = stability_margins(LOOP, corner)
    f = np.logspace(3, 8, 600)
    L = open_loop_gain(f, LOOP, corner)

    # (b) measured suppression, full PDH curve and assumed noise
    x = noise.generate(N, dt, psd, np.random.default_rng(4))
    res = simulate(x, LOOP, corner, disc)
    fm, hm = noise.estimate_transfer(x[SKIP:], res["delta"][SKIP:], dt, nperseg=2**14)
    fm, hm = fm[1:], hm[1:]

    # (c) acquisition from an offset
    n_c = int(1.5e-3 / dt)
    xc = OFFSET + noise.generate(n_c, dt, psd, np.random.default_rng(5))
    engage = int(ENGAGE_T / dt)
    rc = simulate(xc, LOOP, corner, disc, engage=engage)
    window = int(1e-6 / dt)
    avg = np.convolve(rc["delta_cav"], np.ones(window) / window, mode="valid")
    t_avg = (np.arange(avg.size) + window / 2) * dt

    fig = plt.figure(figsize=(10.5, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.loglog(f, np.abs(L), color=style.INK)
    ax.axhline(1, color=style.MUTED, lw=0.8)
    ax.axvline(f_u, color=style.MUTED, ls="--", lw=0.8)
    ax.text(f_u * 1.2, 20, f"unity gain {f_u / 1e6:.2f} MHz\nphase margin {margin:.0f}°",
            color=style.TEXT)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Open-loop gain $|L|$")
    ax.set_title("(a)", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    band = (fm > 2e3) & (fm < 3e7)
    ax.semilogx(fm[band], 10 * np.log10(np.abs(hm[band]) ** 2), color=style.INK, lw=1.0,
                label="Simulation")
    ax.semilogx(f, 10 * np.log10(1 / np.abs(1 + L) ** 2), color=style.ACCENT, lw=2.0, ls="--",
                label=r"$1/|1+L|^2$")
    ax.axhline(0, color=style.MUTED, lw=0.8)
    ax.set_xlim(2e3, 3e7)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Noise suppression (dB)")
    ax.set_title("(b)", loc="left")
    ax.legend(loc="lower right")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(t_avg * 1e3, avg / 1e3, color=style.INK, lw=0.8)
    ax.axvline(ENGAGE_T * 1e3, color=style.MUTED, ls="--", lw=0.8)
    ax.text(ENGAGE_T * 1e3 + 0.02, 0.9 * ax.get_ylim()[1], "lock engaged", color=style.TEXT)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Detuning (kHz)")
    ax.set_title("(c)", loc="left")

    after = rc["delta_cav"][engage + int(50e-6 / dt):]
    print(f"unity-gain frequency {f_u / 1e6:.2f} MHz, phase margin {margin:.1f} deg")
    print(f"locked (after 50 us): mean {np.mean(after):.0f} Hz, rms {np.std(after) / 1e3:.1f} kHz")

    out = ROOT / "figures" / "fig5_loop.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
