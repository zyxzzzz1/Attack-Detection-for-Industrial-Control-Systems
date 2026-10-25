"""Shared chart style (used by all main-report figures, colors consistent with scripts/make_extension_report.py).

Colors come from the dataviz reference palette (light): categorical colors are used in a fixed order (blue, orange, teal), never cycled into new colors;
magnitudes use a single hue (blue) from light to dark; text uses only the ink color, never a series color.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # Categorical colors: slots 1~3
BLUE = SERIES[0]
EVENT_FILL = "#f0efec"   # Background of annotated intervals (neutral gray, not taking a series color)
EVENT_EDGE = "#898781"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],  # The .ttc is registered in matplotlib as JP but contains Simplified Chinese glyphs
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "savefig.dpi": 160, "legend.frameon": False, "font.size": 9,
    })


def clean_axes(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
