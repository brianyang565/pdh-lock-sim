# PDH laser-lock simulation

A Python simulation of a Pound–Drever–Hall (PDH) lock of an external-cavity diode laser to a Fabry–Pérot reference cavity, built from the parameters of a working Rydberg-atom experiment and compared with the noise measured there.

**Summary of results**

- A round-trip simulation of the cavity field reproduces the analytic PDH error signal to $`10^{-12}`$, and shows that the cavity responds to laser frequency noise as a single low-pass filter at its half-width (375 kHz) to within 0.1 % up to 1 MHz.
- A closed-loop model built on this result reproduces the analytic noise suppression $`1/|1+L|^2`$ and the servo bump, acquires lock from any detuning within the modulation frequency (±20 MHz), and locks falsely at 32.3 MHz from beyond it.
- The residual noise is lowest at a proportional gain of 2.5, well below the instability at 6.0, because additional gain enlarges the servo bump faster than it reduces the low-frequency noise.
- The measured frequency noise of the experiment's cavity-transmitted light implies a residual detuning of about 160 kHz rms, 45 % of the cavity half-width. Fits of the full nonlinear simulation to this spectrum place the unity-gain frequency at 560–620 kHz, and show that the measured level is consistent either with a laser 17 times noisier than assumed or with technical noise of about 150 Hz/√Hz at the error-signal point. The two explanations predict spectra that differ by a factor of 18 at 1 kHz, which identifies the measurement that would distinguish them.

## How the project is organized

The simulation follows a laser-frequency fluctuation from the laser, through the cavity and the servo electronics, and back to the laser. It is built in four steps, and each step is checked against an independent calculation before the next step uses it.

| Step | Physics | Library file | Figures |
|---|---|---|---|
| 1 | Reflection from the reference cavity | `pdh_sim/cavity.py` | Fig. 1 |
| 2 | The PDH error signal | `pdh_sim/pdh.py` | Figs. 2, 3 |
| 3 | The cavity as a filter, laser noise, the feedback loop | `pdh_sim/pdh.py`, `noise.py`, `loop.py` | Figs. 4, 5 |
| 4 | Noise spectra, limits of the lock, comparison with measurement | `pdh_sim/loop.py`, `noise.py` | Figs. 6, 7, 8 |

The library `pdh_sim/` contains the physics and draws nothing. Each script in `scripts/` produces one figure, and `run_checks.py` runs the 29 checks in `tests/`.

## Reference cavity

| Quantity | Value | Source |
|---|---|---|
| Free spectral range | 1.5 GHz | experiment |
| Finesse | 2000 | experiment |
| Linewidth (FWHM) $`\delta\nu`$ | 750 kHz | FSR / finesse |
| Mirror reflectivity $`R`$ | 0.99843 | from finesse |
| Photon lifetime | 212 ns | $`1/(2\pi\,\delta\nu)`$ |
| Field time constant $`\tau`$ | 424 ns | $`1/(\pi\,\delta\nu)`$ |
| PDH modulation frequency $`\Omega`$ | 20 MHz | experiment (fixed by the servo electronics) |
| Modulation depth $`\beta`$ | 1.08 rad | **assumed**: maximizes $`J_0 J_1`$ |

The modulation depth of the experiment is not recorded, so $`\beta = 1.08`$ rad, the value that maximizes the error-signal slope, is used throughout. It changes the overall size of the error signal but not its shape near resonance, and it does not affect the loop results of Steps 3 and 4. All default parameters, with the source of each, are collected in `pdh_sim/config.py`.

## Step 1: cavity response

The reflection coefficient of a lossless cavity with identical mirrors of amplitude reflectivity $`r`$ is

```math
F(\delta) = \frac{r\,(e^{i\phi} - 1)}{1 - r^2 e^{i\phi}}, \qquad \phi = \frac{2\pi\delta}{\mathrm{FSR}},
```

where $`\delta`$ is the laser detuning from resonance (Black, Am. J. Phys. **69**, 79 (2001)). The mirror reflectivity is obtained by inverting the finesse definition $`\mathcal{F} = \pi\sqrt{R}/(1-R)`$.

![Cavity response](figures/fig1_cavity_response.png)

**(a)** The reflected power vanishes on resonance because identical lossless mirrors make the cavity impedance matched. The width of the dip, measured numerically without assuming a line shape, is 750.0 kHz, which agrees with FSR / finesse. **(b)** The reflection phase changes by $`2\pi`$ across the resonance. Because $`|F|`$ passes through zero at $`\delta = 0`$, the phase jumps by $`\pi`$ there, and the remaining $`\pi`$ accumulates over roughly one linewidth on either side. The PDH technique converts this phase excursion into an error signal. **(c)** At the modulation sideband frequencies $`\pm\Omega`$ the cavity reflects more than 99.9 % of the power. Since $`\Omega`$ is about 27 linewidths, the sidebands return from the cavity unchanged and act as a fixed phase reference for the carrier.

## Step 2: PDH error signal

### Analytic calculation

The phase-modulated laser field is a comb of sidebands at $`\delta + n\Omega`$ with amplitudes $`J_n(\beta)`$, and sideband $`n`$ is reflected with amplitude $`a_n = J_n(\beta)\,F(\delta + n\Omega)`$. The reflected power therefore oscillates at $`\Omega`$, and mixing the photodiode signal with a local oscillator $`\cos(\Omega t + \theta)`$ and averaging gives the error signal

```math
c_1 = \sum_n a_{n+1}\, a_n^{*}, \qquad \varepsilon(\delta) = \mathrm{Re}\!\left[c_1\, e^{i\theta}\right].
```

Keeping only the carrier and the first-order sidebands reproduces the expression of Black (2001). The code keeps all orders, so that the analytic result can be compared with the simulation at the level of numerical precision. Because $`F(-\delta) = F(\delta)^*`$, $`c_1`$ is an odd function of $`\delta`$, and the error signal vanishes on resonance for every demodulation phase $`\theta`$.

### Time-domain simulation

The simulation advances the intracavity field one round trip ($`1/\mathrm{FSR} = 0.67`$ ns) at a time,

```math
a_k = t\,E_{\mathrm{in},k} + r^2\, a_{k-1}, \qquad E_{\mathrm{ref},k} = -r\,E_{\mathrm{in},k} + t\,r\,a_{k-1}.
```

Light returns to the input mirror exactly one round trip later, so this recursion is exact for any input sampled at the round-trip interval, including a chirped or phase-modulated laser. A modulation period at 20 MHz spans 75 steps. The photodiode power $`|E_{\mathrm{ref}}|^2`$ is multiplied by the local oscillator and passed through a fourth-order Butterworth low-pass filter, as in the experiment's electronics.

The simulation and the analytic formula share no code apart from the cavity parameters. At fixed detunings, where the mixer output is averaged over whole modulation periods after the cavity has settled, they agree to $`10^{-9}`$ of the incident power.

![Error signal](figures/fig2_error_signal.png)

**(a)** A simulated scan at 25 MHz/ms overlaid on the analytic error signal, showing the central feature and the two sideband features at $`\pm\Omega = \pm 20`$ MHz. This is the trace seen on an oscilloscope while the laser is scanned across the cavity. **(b)** Near resonance the simulated trace lags the analytic curve. The lag is the sum of two delays of nearly equal size: the group delay of the 1 MHz low-pass filter (416 ns), and the time with which the intracavity field follows the laser, $`\tau = 1/(\pi\,\delta\nu) = 424`$ ns. At this sweep rate the combined 840 ns corresponds to 21 kHz of detuning. Halving the sweep rate halves the deviation, which shows that the lag is dynamical and not an error of the calculation.

The 424 ns delay in panel (b) is the thread that connects everything after this point. The cavity does not respond instantly to a change of the laser frequency; it needs about 424 ns to catch up. During a slow scan this produces only a small lag, but laser noise consists of much faster changes, and for those the delay matters a great deal.

### Why 20 MHz works well

![Modulation regimes](figures/fig3_modulation_regimes.png)

The scope trace in Fig. 2(a) has three well-separated features: the steep central slope and two small features at ±20 MHz, where the sidebands cross the resonance. They are well separated because the modulation frequency is 27 times the cavity linewidth. Fig. 3 shows what the trace would look like if the ratio $`\Omega/\delta\nu`$ were smaller. As the ratio decreases, the three features merge into a single broad wiggle, and the central slope becomes shallower, because the sidebands are then partly transmitted through the cavity instead of being reflected as a clean reference.

The modulation frequency is fixed by the servo electronics, so this figure is not a proposal to change it. It shows that the experiment sits in the favorable regime: at $`\Omega/\delta\nu = 27`$ the slope equals the largest possible value, $`4J_0 J_1/\delta\nu`$, to within 0.1 %, and a steeper slope gives the servo a stronger signal for a given frequency error.

## Step 3: closing the loop

### From a slow scan to fast noise: the cavity as a filter

On the scope, the laser is scanned slowly through resonance, and the error signal traces out the curve of Fig. 2. When the laser is locked, it is not scanned; instead, its frequency jitters around resonance because of noise, and some of that jitter is fast. The first question for the lock is therefore whether the error signal reports a fast jitter as faithfully as it reports a slow scan.

The answer comes from the 424 ns delay. The light reflected from the cavity has two parts: light that bounces directly off the input mirror, and light that has been stored in the cavity and leaks back out. The stored light behaves like a running average of the laser over the last 424 ns. The error signal compares the light arriving now with this average.

- If the laser frequency changes slowly, the stored light keeps up, and the error signal reports the full frequency offset, exactly as in the scan.
- If the laser frequency wobbles faster than the stored light can follow, the average washes the wobble out, and the error signal reports only part of it, and late.

The crossover lies where the wobble advances by one radian during one storage time, at $`f_c = 1/(2\pi\tau) = \delta\nu/2 = 375`$ kHz. Mathematically, the fraction of a wobble at rate $`f`$ that the error signal reports, together with its delay, is

```math
H_c(f) = \frac{1}{1 + i f/f_c}, \qquad f_c = \frac{\delta\nu}{2} = 375\ \text{kHz},
```

which has the same form as an RC low-pass filter. Its magnitude is the fraction reported, and its phase is the delay expressed as an angle.

| Wobble rate | Fraction reported $`\lvert H_c \rvert`$ | Phase lag | Report arrives late by |
|---|---|---|---|
| 10 kHz | 1.00 | 1.5° | 424 ns |
| 100 kHz | 0.97 | 15° | 414 ns |
| 375 kHz | 0.71 | 45° | 333 ns |
| 1 MHz | 0.35 | 69° | 193 ns |
| 4 MHz | 0.09 | 85° | 59 ns |

![Cavity FM response](figures/fig4_cavity_fm_response.png)

Fig. 4 tests this picture directly. In the full optical simulation the laser frequency is made to wobble sinusoidally by a small amount, and the size and delay of the resulting wobble in the error signal are measured, one wobble rate at a time from 50 kHz to 4 MHz. The simulation (markers) follows $`H_c(f)`$ (line) to within 0.1 % up to 1 MHz and to 1 % at 4 MHz.

This result has a practical consequence for the code. Inside the feedback loop the cavity can be replaced by this one filter followed by the error curve of Fig. 2, instead of following the light round trip by round trip. The loop can then advance in 10 ns steps rather than 0.67 ns steps, so milliseconds of locked operation take seconds to simulate.

### The laser noise that the lock must remove

When the laser is locked, the error signal on the scope is not perfectly flat; it jitters, and that jitter is the laser's frequency noise as seen through the lock. To simulate it, the code needs a recipe for random frequency fluctuations. The recipe is a noise spectrum, $`S_\nu(f)`$ in Hz²/Hz, which states how much frequency jitter the laser has at each rate $`f`$. It is the quantity a spectrum analyzer displays when it is connected to a frequency-noise signal.

The model contains two kinds of noise:

```math
S_\nu(f) = h_0 + \frac{h_1}{f}.
```

White noise $`h_0`$ has the same strength at every rate and represents the fast, intrinsic noise of the diode. Flicker noise $`h_1/f`$ grows towards low frequency and represents slow technical drifts. The manufacturer quotes a typical linewidth below 100 kHz without stating an averaging time, and the free-running noise of the laser in the experiment has not been measured, so both terms are assumptions:

| Quantity | Value | Basis |
|---|---|---|
| Intrinsic (Lorentzian) linewidth $`\pi h_0`$ | 10 kHz | **assumed** |
| Flicker coefficient $`h_1`$ | $`4.9 \times 10^8`$ Hz² | **assumed**: total linewidth 100 kHz over 1 ms |

The linewidth is computed from the spectrum by the β-separation-line method of Di Domenico *et al.* (Appl. Opt. **49**, 4801 (2010)). The β in this name is unrelated to the modulation depth.

### The loop: what the servo does to a fluctuation

Engaging the lock on the scope collapses the swept error signal to a small jitter around zero. The servo achieves this by measuring the error, computing a correction, and changing the diode current. A frequency fluctuation travels around the following loop:

```
laser noise ──► (+) ──► cavity ──► PDH error ──► delay ──► PI controller ──► diode current ──┐
                 ▲     (375 kHz     signal      (100 ns)   (gain knobs)      (3 MHz)        │
                 │      filter)                                                             │
                 └──────────────────────────── correction ◄─────────────────────────────────┘
```

The PI controller contains the two knobs of the servo. The proportional term pushes back in proportion to the present error, with gain $`G`$. The integral term accumulates the error over time, so even a small constant error grows the correction until the error vanishes; this term removes slow drift and pulls the laser in from far away.

The essential limitation is that every correction arrives late. The cavity adds up to 424 ns, the electronics and cables about 100 ns, and the current actuator about 50 ns. Whether a delay of roughly half a microsecond matters depends on how fast the noise is:

| Noise rate | One period of the noise | What happens |
|---|---|---|
| 10 kHz | 100 µs | The correction arrives essentially on time and cancels the noise. |
| 1.6 MHz | 625 ns | The correction arrives about half a period late, when the noise has already reversed, so it pushes the same way as the noise and enlarges it. |
| 5 MHz | 200 ns | The servo's response has become too small to matter, and the noise passes untouched. |

This table contains the whole behavior of the lock, and the mathematics below only makes it quantitative. A noise component at rate $`f`$ that goes once around the loop returns scaled in size and shifted in time. That round-trip factor is the open-loop gain,

```math
L(f) = G\left(1 + \frac{f_{\mathrm{int}}}{i f}\right) \cdot \frac{1}{1 + i f/f_c} \cdot \frac{1}{1 + i f/f_{\mathrm{act}}} \cdot e^{-2\pi i f \tau_d},
```

whose four factors are the controller, the cavity, the actuator and the delay. The detuning is the free-running noise plus the correction, and the correction is $`-L`$ times the detuning, so

```math
\delta = \nu_{\mathrm{free}} - L\,\delta \quad\Longrightarrow\quad \delta = \frac{\nu_{\mathrm{free}}}{1 + L}.
```

When $`|L|`$ is large the noise is reduced by the factor $`|L|`$. When $`|L|`$ is small the noise is unchanged. When $`|L| \approx 1`$ and the delays have rotated $`L`$ by nearly 180°, the denominator $`1 + L`$ becomes small and the noise is amplified; this is the servo bump. If $`L = -1`$ exactly, the loop oscillates on its own, which is the ringing seen on the scope when the gain is turned up too far.

| Quantity | Value | Basis |
|---|---|---|
| Cavity corner $`f_c`$ | 375 kHz | $`\delta\nu/2`$ |
| Actuator bandwidth $`f_{\mathrm{act}}`$ | 3 MHz | B1047 headboard, MOGLabs manual |
| Loop delay $`\tau_d`$ | 100 ns | 40 ns headboard (manual) + 60 ns **assumed** for servo and cables |
| Proportional gain $`G`$ | 3 | chosen |
| Integrator corner $`f_{\mathrm{int}}`$ | 50 kHz | chosen |

![Loop](figures/fig5_loop.png)

**(a)** The open-loop gain. It falls to 1 at 1.0 MHz; at that frequency the delays have rotated the phase by 127° (69° from the cavity, 36° from the electronics delay, 18° from the actuator and 3° from the integrator), leaving a phase margin of 53° before the 180° at which the loop would oscillate. The loop can act above the 375 kHz cavity corner because above that corner the cavity response falls as $`1/f`$, which is the gradual roll-off a stable loop needs. **(b)** The noise suppression measured in the time-domain simulation agrees with $`1/|1+L|^2`$ across the whole band. Below 100 kHz the noise is reduced by 10 to 30 dB, near 1.6 MHz it is amplified by 7 dB (the servo bump), and above a few megahertz the loop has no effect. **(c)** The equivalent of engaging the lock on the scope: the laser starts 300 kHz from resonance, and when the loop is switched on at 0.5 ms it is pulled onto resonance within a few microseconds and stays there.

## Step 4: performance, limits and comparison with measurement

### What noise remains when the laser is locked

![Spectra](figures/fig6_spectra.png)

Fig. 6 is the spectrum-analyzer view of the lock: the noise spectrum of the free-running laser (blue), of the locked laser (orange), and of the light transmitted through the cavity (green). Because each noise rate passes through the loop independently, the locked spectra are the free-running spectrum multiplied by the suppression factors of Step 3:

```math
S_\delta(f) = \frac{S_{\mathrm{free}}(f)}{|1 + L(f)|^2}, \qquad S_{\mathrm{trans}}(f) = \frac{|H_c(f)|^2\, S_{\mathrm{free}}(f)}{|1 + L(f)|^2}.
```

The figure shows three regions, in the same order as the table in Step 3. Below about 100 kHz the lock removes the noise strongly, by up to a factor of 10⁴ in power. Near 1.6 MHz the servo bump raises it. Above a few megahertz the locked laser is as noisy as the free-running one, because the corrections arrive too late. The transmitted light (green) is much quieter at high frequency, because the cavity itself filters out fast noise that the servo cannot catch. This is why the experiment derives its reference light from the cavity transmission.

The dotted line is the β-separation line. Noise above it broadens the laser line, and noise below it only adds weak wings. The free-running spectrum lies above the line at low frequency and gives a 100 kHz linewidth. The locked spectrum lies below it everywhere, so the locked line, relative to the cavity, is a narrow carrier less than 1 kHz wide. The simulated cavity is a perfect reference, so this width does not include the frequency noise of the cavity itself.

### The limits of the lock

![Lock limits](figures/fig7_lock_limits.png)

**(a) The gain knob.** Turning up the proportional gain makes the loop stronger at every rate, which improves the suppression at low frequency. It also moves the servo bump closer to the point of instability, because the faster the loop acts, the larger the rotation the delays produce. The residual noise therefore falls, reaches a minimum of 38 kHz rms at $`G = 2.5`$, and rises again. At $`G = 6.0`$ the loop oscillates; beyond it, the simulated residual grows because the lock rings, and the ringing is limited only by the saturation of the error signal.

**(b, c) Engaging the lock away from resonance.** Panel (c) is the scope trace of Fig. 2 again, shaded between −20 and +20 MHz. Within that range the error signal, although small away from the central slope, always has the sign that pushes the laser towards resonance, and the integrator accumulates it until the laser arrives. Panel (b) shows the result: started 1, 10 or 19 MHz away, the laser is pulled onto resonance, in 9 µs from 1 MHz and in about 0.8 ms from 19 MHz. Started 21 MHz away, beyond the sideband feature, the error signal has the wrong sign, and the laser is pushed away until it settles at 32.3 MHz. There the error signal crosses zero again with the slope that holds the laser in place, so the servo is stably locked to the wrong frequency. This false lock point arises from the beat between the first- and second-order sidebands.

**Noise tolerance.** Raising the intrinsic linewidth from 10 kHz to 1 MHz increases the residual noise in proportion to the square root of the white-noise level, in agreement with linear theory to within 3 %. Within the model the lock does not break even at a 10 MHz intrinsic linewidth, because every excursion stays inside the ±20 MHz pull-in range. At such noise levels the laser frequency changes faster than the regime verified in Fig. 4, so no loss-of-lock threshold is quoted.

### Comparison with the measured noise of the experiment

The thesis describes the frequency noise of the cavity-transmitted 780 nm light by

```math
S(f) = h_0\, e^{-Af}, \qquad h_0 = 0.026(5)\ \text{MHz}, \quad A = 1.0(9)\ \text{MHz}^{-1}.
```

The parameters come from the noise spectrum of the transmitted light recorded on a photodiode, and independently from the interference visibility of the stored excitation as a function of storage time. A photodiode on the transmitted light responds to the laser frequency relative to the cavity, after the cavity filter, which is the green curve of Fig. 6. The β-line linewidth of this spectrum is 78 kHz, which reproduces the linewidth quoted in the thesis.

The measured level is 15 to 60 times higher than the green curve of Fig. 6. The loop itself has been verified, so the difference must lie in its inputs. Two explanations are tested:

- **H1, a noisier laser.** The free-running laser has more noise than assumed, by a factor $`k`$.
- **H2, noise at the error signal.** Electronic noise from the photodetector and amplifiers, or residual amplitude modulation, adds a fake signal to the error voltage. The servo cannot tell this fake signal from a real frequency error, so it "corrects" the laser in response to it and writes the noise onto the light. The same algebra as in Step 3 gives a transfer factor $`L/(1+L)`$, which is close to 1 wherever the loop has gain: the servo transfers this noise to the laser at full strength instead of suppressing it.

![Thesis comparison](figures/fig8_thesis_comparison.png)

The measured spectrum corresponds to a residual detuning of $`\sqrt{h_0/A} \approx 160`$ kHz rms, about 45 % of the 375 kHz half-width. At this amplitude the laser spends time on the curved part of the error signal, near its turning points, where the error signal is weaker than the straight-line approximation assumes. Each hypothesis is therefore fitted to the measured spectrum between 10 kHz and 3 MHz using the full simulation, which keeps that curvature, with the loop gain as a second free parameter.

| | H1: noisier laser | H2: error-point noise |
|---|---|---|
| Fitted noise | $`k = 17`$ | 148 Hz/√Hz |
| Loop gain $`G`$ | 2.0 | 1.8 |
| Unity-gain frequency, phase margin | 620 kHz, 82° | 560 kHz, 88° |
| rms misfit | factor 1.18 | factor 1.23 |

Both explanations reproduce the measurement equally well, and both require a loop with a unity-gain frequency near 600 kHz and a large phase margin, which explains why the measured spectrum shows no pronounced servo bump. They imply different physics. Under H1 the free-running laser would have a linewidth of about 535 kHz over 1 ms, more than five times the manufacturer's typical value. Under H2 the noise at the error signal would exceed the shot-noise limit of the PDH signal (Black 2001: 0.003 Hz/√Hz for 1 mW of carrier power, 0.03 Hz/√Hz for 10 µW) by more than three orders of magnitude, so it would be technical in origin.

The two explanations disagree below 10 kHz. At those rates the servo is strongest. Under H1 it removes most of the laser noise, so the spectrum of the transmitted light falls towards low frequency (blue). Under H2 the servo itself writes the electronic noise onto the laser, so the spectrum stays flat (orange). At 1 kHz the predictions differ by a factor of 18, so a measurement of the transmitted-light spectrum between 1 and 10 kHz would decide between them.

**Limitations.** The simulation treats the cavity as a perfect frequency reference. Fluctuations of the cavity length would not appear on a photodiode that detects the transmitted light, since the laser follows the cavity, but they would appear in a measurement with the atoms. The noise floor of the photodiode measurement is not modeled, and $`A`$ is poorly constrained: repeating the linear-theory fit with $`A = 0.5`$ and 2 MHz⁻¹ instead of 1 MHz⁻¹ moves the fitted unity-gain frequency to about 660 and 40 kHz, respectively.

## Running

The code needs numpy, scipy and matplotlib, all of which are included in Anaconda.

**In Spyder.** Download the repository (green **Code** button → **Download ZIP**) and unzip it, or clone it. Open `run_checks.py` and press F5; each check prints PASS or FAIL with a one-line description. To regenerate a figure, open the corresponding script in `scripts/` and press F5; the figure appears in the Plots pane and is saved to `figures/`. Most figures take a few seconds; Figure 2 takes about 15 s, Figure 7 about 20 s, and Figure 8, which fits the full simulation to the measured spectrum, about 1–2 minutes.

**From a terminal.** The same checks also run under pytest:

```
pip install -r requirements.txt
python run_checks.py      # or: pytest
python scripts/fig1_cavity_response.py
```

## References

- E. D. Black, "An introduction to Pound–Drever–Hall laser frequency stabilization," Am. J. Phys. **69**, 79 (2001).
- G. Di Domenico, S. Schilt and P. Thomann, "Simple approach to the relation between laser frequency noise and laser line shape," Appl. Opt. **49**, 4801 (2010).
- MOGLabs, [Cateye Laser CEL, CEX and CEF manual, rev. 2.08](https://www.moglabs.com/products/lasers/tunable-lasers/cel/MOGLabs_CEL_manual.pdf).
