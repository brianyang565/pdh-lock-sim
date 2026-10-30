"""Default parameters shared by the tests and the figure scripts.

Each value is marked as coming from the experiment (thesis), from the
manufacturer's documentation, or as an assumption made for the
simulation. Assumed values should be replaced by measured ones when these
are available.
"""

from .loop import LoopParams
from .noise import lorentzian_fwhm_to_h0

# PDH modulation ---------------------------------------------------------
OMEGA_MOD = 20e6        # Hz, thesis Sec. 2.4.2
BETA = 1.08             # rad, assumed: maximises J0 J1 and hence the slope

# Free-running laser noise (MOGLabs cat-eye laser) ----------------------
# The manufacturer quotes a typical linewidth below 100 kHz without an
# averaging time. The model below assumes a 10 kHz intrinsic (Lorentzian)
# linewidth and a flicker level chosen so that the beta-line linewidth
# over 1 ms equals 100 kHz.
INTRINSIC_LINEWIDTH = 10e3                       # Hz, assumed
H0 = lorentzian_fwhm_to_h0(INTRINSIC_LINEWIDTH)  # Hz^2/Hz, white frequency noise
H1 = 4.89e8                                      # Hz^2, flicker: gives 100 kHz at 1 ms

# Feedback loop ---------------------------------------------------------
LOOP = LoopParams(
    gain=3.0,       # proportional loop gain, chosen for ~1 MHz unity-gain frequency
    f_int=50e3,     # Hz, PI corner, chosen
    f_act=3e6,      # Hz, B1047 headboard current-modulation bandwidth (MOGLabs manual)
    delay=100e-9,   # s: 40 ns headboard phase delay (manual) + 60 ns assumed for servo and cables
    dt=10e-9,       # s, simulation step
)

# Measured noise of the cavity-transmitted 780 nm light (thesis Sec. 3) --
# S(f) = h0 exp(-A f), from the photodiode noise spectrum and, independently,
# from the fit of Eq. 3.14 to the interference visibility (Fig. 3.10).
H0_THESIS = 0.026e6     # Hz^2/Hz  (0.026(5) MHz)
A_THESIS = 1.0e-6       # 1/Hz     (1.0(9) MHz^-1)
A_THESIS_RANGE = (0.1e-6, 1.9e-6)
WAVELENGTH = 780e-9     # m
