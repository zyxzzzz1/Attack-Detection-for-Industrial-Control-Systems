"""图表的公共样式（主线报告图共用，与 scripts/make_extension_report.py 的配色一致）。

配色取自 dataviz 参考调色板（浅色）：类别色按固定顺序使用（蓝、橙、青），不循环生成新颜色；
数值大小用单一色相（蓝）由浅到深；文字只用墨色，不用系列色。
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # 类别色：槽位 1~3
BLUE = SERIES[0]
EVENT_FILL = "#f0efec"   # 标注区间的底色（中性灰，不占用系列色）
EVENT_EDGE = "#898781"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],  # .ttc 在 matplotlib 中登记为 JP，含简体字形
        "axes.unicode_minus": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "savefig.dpi": 160, "legend.frameon": False, "font.size": 9,
    })


def clean_axes(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
