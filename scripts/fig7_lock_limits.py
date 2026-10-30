"""Figure 7: limits of the lock.

(a) Gain sweep. Residual rms detuning seen through the cavity, from linear
    theory (line, shown only where the loop is stable) and from the
    time-domain simulation (markers), as a function of the proportional
    gain G. The phase margin reaches zero at the marked gain.
(b) Lock acquisition from different initial detunings. The laser is pulled
    in from any offset below the modulation frequency; from beyond it, the
    loop drives the laser to the false lock point.
(c) The error signal that explains (b): its sign is correct for pull-in
    everywhere within +-Omega, and it crosses zero again with the
    stabilising slope between Omega and 2 Omega.

In Spyder, open this file and press F5 (about 20 s).
"""

import dataclasses
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import style  # noqa: E402
from pdh_sim import noise, thesis_cavity  # noqa: E402
from pdh_sim.config import BETA, H0, H1, LOOP, OMEGA_MOD  # noqa: E402
from pdh_sim.loop import build_discriminator, locked_spectra, simulate, stability_margins  # noqa: E402
from pdh_sim.pdh import error_signal, optimal_phase  # noqa: E402

GAINS_SIM = [0.5, 1, 1.5, 2, 3, 4, 5, 5.5, 6.5, 7, 8]
OFFSETS = [1e6, 10e6, 19e6, 21e6]  # initial detunings for panel (b)


def false_lock_point(cav, phase):
    """First zero crossing beyond Omega at which the error has the stabilising (positive) slope."""
    d = np.linspace(1.02 * OMEGA_MOD, 2 * OMEGA_MOD, 200001)
    e = error_signal(cav, d, OMEGA_MOD, BETA, phase)
    idx = np.nonzero((e[:-1] < 0) & (e[1:] >= 0))[0]
    i = idx[0]
    return optimize.brentq(lambda x: error_signal(cav, x, OMEGA_MOD, BETA, phase), d[i], d[i + 1])


def main():
    style.apply()
    cav = thesis_cavity()
    corner = cav.linewidth / 2
    dt = LOOP.dt
    phase = optimal_phase(cav, OMEGA_MOD, BETA)
    disc = build_discriminator(cav, OMEGA_MOD, BETA, phase, span=4 * OMEGA_MOD, points=400001)
    psd = lambda f: noise.white_and_flicker_psd(f, H0, H1)  # noqa: E731

    # ---- (a) gain sweep ------------------------------------------------
    g_crit = optimize.brentq(
        lambda g: stability_margins(dataclasses.replace(LOOP, gain=g), corner)[1], 4.0, 8.0)
    fg = np.logspace(2, np.log10(0.5 / dt), 20001)
    g_th = np.linspace(0.3, 0.97 * g_crit, 80)
    rms_th = []
    for g in g_th:
        _, sc = locked_spectra(fg, psd(fg), 0.0, dataclasses.replace(LOOP, gain=g), corner, discrete=True)
        rms_th.append(np.sqrt(integrate.trapezoid(sc, fg)))
    x = noise.generate(2**18, dt, psd, np.random.default_rng(3))
    rms_sim = [np.std(simulate(x, dataclasses.replace(LOOP, gain=g), corner, disc)["delta_cav"][2**14:])
               for g in GAINS_SIM]

    # ---- (b) acquisition -----------------------------------------------
    n = int(6e-3 / dt)
    xb = noise.generate(n, dt, psd, np.random.default_rng(8))
    window = int(10e-6 / dt)
    traces = []
    for off in OFFSETS:
        d = simulate(off + xb, LOOP, corner, disc)["delta_cav"]
        avg = np.convolve(d, np.ones(window) / window, mode="valid")[::100]
        traces.append(avg)
    t_b = (np.arange(traces[0].size) * 100 + window / 2) * dt
    f_false = false_lock_point(cav, phase)

    # ---- (c) error signal ----------------------------------------------
    d_c = np.linspace(-45e6, 45e6, 20001)
    e_c = error_signal(cav, d_c, OMEGA_MOD, BETA, phase)

    fig = plt.figure(figsize=(11, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(g_th, np.array(rms_th) / 1e3, color=style.ACCENT, lw=2.0, label="Linear theory")
    ax.plot(GAINS_SIM, np.array(rms_sim) / 1e3, "o", ms=6, mfc="white", mec=style.INK, mew=1.5,
            label="Simulation")
    ax.axvline(g_crit, color=style.MUTED, ls="--", lw=0.8)
    ax.text(g_crit * 1.02, 150, f"phase margin 0\nG = {g_crit:.1f}", color=style.TEXT)
    i_min = int(np.argmin(rms_th))
    ax.annotate(f"minimum {rms_th[i_min] / 1e3:.0f} kHz\nat G = {g_th[i_min]:.1f}",
                xy=(g_th[i_min], rms_th[i_min] / 1e3), xytext=(0.6, 110),
                arrowprops=dict(arrowstyle="->", color=style.MUTED, lw=0.9), color=style.TEXT)
    ax.set_xlabel("Proportional gain G")
    ax.set_ylabel("Residual detuning, rms (kHz)")
    ax.set_ylim(0, 200)
    ax.set_title("(a)", loc="left")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    for tr, off, colour in zip(traces, OFFSETS, style.SERIES + [style.INK]):
        ax.plot(t_b * 1e3, tr / 1e6, color=colour, lw=1.4, label=f"start {off / 1e6:.0f} MHz")
    ax.axhline(OMEGA_MOD / 1e6, color=style.MUTED, ls="--", lw=0.8)
    ax.axhline(f_false / 1e6, color=style.MUTED, ls=":", lw=0.9)
    ax.text(3.0, OMEGA_MOD / 1e6 + 0.8, r"$\Omega$ = 20 MHz", ha="center", color=style.TEXT)
    ax.text(t_b[-1] * 1e3, f_false / 1e6 + 0.8, f"false lock {f_false / 1e6:.1f} MHz", ha="right",
            color=style.TEXT)
    ax.set_xlabel("Time after the lock is engaged (ms)")
    ax.set_ylabel("Detuning, 10 µs average (MHz)")
    ax.set_ylim(-2, 36)
    ax.set_title("(b)", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.3), fontsize=8)

    ax = fig.add_subplot(gs[1, :])
    ax.axvspan(-OMEGA_MOD / 1e6, OMEGA_MOD / 1e6, color="#eef3fa", zorder=0)
    ax.plot(d_c / 1e6, e_c, color=style.INK)
    ax.axhline(0, color=style.MUTED, lw=0.8)
    for s in (-1, 1):
        ax.plot(s * f_false / 1e6, 0, "o", ms=8, mfc="white", mec=style.ACCENT, mew=1.8)
    ax.text(-10, -0.25, "capture range  $|\\delta| < \\Omega$", ha="center", color=style.TEXT)
    ax.text(f_false / 1e6, 0.05, "false lock", ha="center", color=style.TEXT)
    ax.text(-f_false / 1e6, 0.05, "false lock", ha="center", color=style.TEXT)
    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel(r"Error signal ($P_0$)")
    ax.set_title("(c)", loc="left")

    print(f"critical gain (phase margin 0): {g_crit:.2f}")
    print(f"minimum residual {rms_th[i_min] / 1e3:.1f} kHz at G = {g_th[i_min]:.2f}")
    for g, r in zip(GAINS_SIM, rms_sim):
        print(f"  G = {g:4.1f}: simulated rms {r / 1e3:6.1f} kHz")
    print(f"false lock point: {f_false / 1e6:.3f} MHz")
    for off, tr in zip(OFFSETS, traces):
        inside = np.nonzero(np.abs(tr) < 50e3)[0]
        t_acq = t_b[inside[0]] * 1e6 if inside.size else float("nan")
        print(f"  from {off / 1e6:4.0f} MHz: final {tr[-1] / 1e6:6.2f} MHz, "
              f"first within 50 kHz after {t_acq:.0f} us")

    out = ROOT / "figures" / "fig7_lock_limits.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")
    plt.show()


if __name__ == "__main__":
    main()
