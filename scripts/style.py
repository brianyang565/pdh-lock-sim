"""Shared matplotlib style for the project figures."""

import matplotlib.pyplot as plt

INK = "#1f4e79"     # primary series
ACCENT = "#c55a11"  # second series (distinguishable from INK under colour-vision deficiency)
MUTED = "#6b6b6b"   # annotations, reference lines
GRID = "#e3e3e3"
TEXT = "#222222"


def apply():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "legend.frameon": False,
        }
    )

# Categorical series for figures with more than two curves, validated for
# colour-vision deficiency with the dataviz palette checker. The third hue
# has low contrast on white, so curves using it are also labelled directly.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
