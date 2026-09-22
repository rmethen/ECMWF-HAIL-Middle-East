"""Plot operational tropical outgoing-longwave-radiation Hovmoller."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


DATA_FILE = Path("data/hovmoller_olr_raw.npz")
OUT_FILE = Path("output/HOVMOLLER_OLR_LATEST.png")
LEVELS = np.arange(100, 321, 10)
COLORS = [
    "#3b0078", "#57109a", "#6f28b8", "#824bd0", "#766fe0",
    "#5b91e8", "#3aafe5", "#20c9d0", "#2bd3a1", "#56da73",
    "#8ee05a", "#c7e85a", "#edf17a", "#fffbe0", "#ffffff",
    "#ffffff", "#fffbe0", "#ffe8a8", "#ffc75b", "#f77b2e",
    "#e94b2d", "#c92332",
]


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smooth(field: np.ndarray) -> np.ndarray:
    # OLR contains many short-lived cloud clusters.  About 30 hours in time
    # and 5 degrees in longitude suppresses that speckle while retaining the
    # broad eastward-moving MJO envelope.
    time_kernel = gaussian_kernel(2.25)
    lon_kernel = gaussian_kernel(10.0)
    tp = len(time_kernel) // 2
    lp = len(lon_kernel) // 2
    time_data = np.pad(field, ((tp, tp), (0, 0)), mode="edge")
    time_data = np.apply_along_axis(
        lambda x: np.convolve(x, time_kernel, mode="valid"), 0, time_data)
    lon_data = np.pad(time_data, ((0, 0), (lp, lp)), mode="wrap")
    return np.apply_along_axis(
        lambda x: np.convolve(x, lon_kernel, mode="valid"), 1, lon_data)


def main() -> None:
    data = np.load(DATA_FILE)
    lon = np.asarray(data["longitude"], dtype=float)
    times = data["time"].astype("datetime64[m]").astype(object)
    olr = smooth(np.asarray(data["olr"], dtype=float))
    start = data["forecast_start"].astype("datetime64[m]").item()
    cycle = str(data["source_cycle"].item())

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, cmap.N)
    fig, ax = plt.subplots(figsize=(10, 11))
    mesh = ax.contourf(lon, times, olr, levels=LEVELS, cmap=cmap,
                       norm=norm, extend="both")
    wet = ax.contour(lon, times, olr, levels=[160, 180, 200],
                     colors="black", linewidths=[1.25, 1.0, 0.7])
    ax.clabel(wet, fmt="%d", fontsize=7, inline=True)
    ax.axhline(start, color="#e99aac", linewidth=5, alpha=0.8)
    ax.text(180, start, "  Begin GFS Forecast", ha="center", va="top",
            fontsize=10, color="black")

    ax.set_xlim(0, 360)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(color="0.5", linewidth=0.35, alpha=0.35)
    ax.set_title("Outgoing Longwave Radiation — Operational", fontsize=17,
                 pad=12, loc="left")
    ax.text(1, 1.015, "[15°S–15°N]", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=13)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.09,
                        fraction=0.05, ticks=np.arange(100, 321, 20))
    cbar.set_label("OLR [W m⁻²]   Purple/green: wet convection | Orange/red: dry suppression",
                   fontsize=10)
    fig.text(0.5, 0.015,
             f"Black contours (160, 180, 200 W m⁻²) emphasize deep convection | GFS {cycle}",
             ha="center", fontsize=9)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
