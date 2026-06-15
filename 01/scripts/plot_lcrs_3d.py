from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "lcrs.txt"
RESULT_DIR = ROOT / "result"
OUTPUT_FILE = RESULT_DIR / "lcrs_3d_views.png"


def load_numeric_rows(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].isdigit():
            rows.append([float(value) for value in parts])
    return np.array(rows, dtype=float)


def spherical_to_cartesian(radius: np.ndarray, theta: np.ndarray, phi: np.ndarray):
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z


def magnitude_to_sizes(magnitude: np.ndarray) -> np.ndarray:
    spread = magnitude.max() - magnitude.min()
    if spread == 0:
        return np.full_like(magnitude, 14.0)
    brightness = (magnitude.max() - magnitude) / spread
    return 10 + 18 * brightness


def style_3d_axes(ax, axis_limit: float):
    ax.set_proj_type("ortho")
    ax.view_init(elev=22, azim=36)
    ax.set_box_aspect((1, 1, 0.85))

    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=4))

    ax.tick_params(labelsize=8, pad=2)
    ax.grid(True, alpha=0.18)

    # Remove pane fills so the data cloud stands out more than the bounding box.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#d0d7de")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D Distribution", pad=10, fontsize=12)


def style_projection_axes(ax, xlabel: str, ylabel: str, axis_limit: float, title: str):
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8, fontsize=11)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(labelsize=8)
    ax.grid(True, color="#b8c4d0", alpha=0.3, linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4af")
    ax.spines["bottom"].set_color("#9aa4af")

def main():
    data = load_numeric_rows(DATA_FILE)
    radius = data[:, 0]
    theta = data[:, 1]
    phi = data[:, 2]
    magnitude = data[:, 3]

    x, y, z = spherical_to_cartesian(radius, theta, phi)
    axis_limit = float(np.max(np.abs(np.concatenate([x, y, z]))))
    sizes = magnitude_to_sizes(magnitude)
    cmap = "cividis_r"
    norm = colors.Normalize(vmin=float(magnitude.min()), vmax=float(magnitude.max()))

    fig = plt.figure(figsize=(15.5, 10.5))
    fig.patch.set_facecolor("#fcfcfb")
    grid = fig.add_gridspec(
        3,
        3,
        width_ratios=(1.35, 1.35, 1.0),
        left=0.04,
        right=0.94,
        top=0.90,
        bottom=0.10,
        wspace=0.28,
        hspace=0.34,
    )

    ax_3d = fig.add_subplot(grid[:, :2], projection="3d")
    scatter = ax_3d.scatter(
        x,
        y,
        z,
        c=magnitude,
        cmap=cmap,
        norm=norm,
        s=sizes,
        alpha=0.78,
        linewidths=0,
    )
    style_3d_axes(ax_3d, axis_limit)

    projections = [
        (fig.add_subplot(grid[0, 2]), x, y, "x", "y", "XY Projection"),
        (fig.add_subplot(grid[1, 2]), x, z, "x", "z", "XZ Projection"),
        (fig.add_subplot(grid[2, 2]), y, z, "y", "z", "YZ Projection"),
    ]

    for ax, horizontal, vertical, xlabel, ylabel, title in projections:
        ax.scatter(
            horizontal,
            vertical,
            c=magnitude,
            cmap=cmap,
            norm=norm,
            s=sizes * 0.85,
            alpha=0.72,
            linewidths=0,
        )
        style_projection_axes(ax, xlabel, ylabel, axis_limit, title)

    colorbar = fig.colorbar(
        scatter,
        ax=[ax_3d, *(ax for ax, *_ in projections)],
        shrink=0.84,
        pad=0.02,
    )
    colorbar.set_label("Absolute Magnitude (brighter -> darker)")
    fig.suptitle("Las Campanas Redshift Survey Galaxies", fontsize=20, y=0.965)
    fig.text(
        0.5,
        0.035,
        "Radius is approximated by recession velocity; point size increases with intrinsic brightness.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=220, bbox_inches="tight")
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
