"""Plot 850-hPa equatorial zonal-wind anomalies as a Hovmoeller diagram."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


DATA_FILE = Path("data/hovmoller_u850_anomaly.npz")
OUT_FILE = Path("output/HOVMOLLER_U850_ANOMALY_LATEST.png")

# Fixed scale: never auto-rescale between runs.  This makes extremes directly
# comparable and preserves the visual convention in the reference product.
LEVELS = np.arange(-15, 16, 1)
COLORS = [
    "#f3bad8", "#ed8cdb", "#df51e2", "#c800e8", "#9700df",
    "#6400d4", "#2600c7", "#0020b8", "#0054d6", "#2b8be8",
    "#63baf0", "#98dff4", "#d9f6fa", "#ffffff", "#ffffff",
    "#fff56b", "#ffe32f", "#ffc21c", "#ff9812", "#ff6c08",
    "#ff3505", "#ed0905", "#c90000", "#9e0000", "#781400",
    "#613000", "#db7084", "#ee9dad", "#f3c1ca", "#f7dde3",
]


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Return a compact normalized Gaussian kernel."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smooth_hovmoller(field: np.ndarray) -> np.ndarray:
    """Suppress grid-scale noise while retaining propagating wind envelopes.

    GFS is sampled every 6 hours on a 0.25-degree grid.  A roughly 24-hour
    temporal filter and 3-degree cyclic longitudinal filter match the clean
    appearance of operational Hovmoeller products without moving the centres
    of the westerly/easterly signals.
    """
    time_kernel = gaussian_kernel(1.5)
    lon_kernel = gaussian_kernel(5.0)

    time_pad = len(time_kernel) // 2
    padded_time = np.pad(field, ((time_pad, time_pad), (0, 0)), mode="edge")
    smoothed_time = np.apply_along_axis(
        lambda row: np.convolve(row, time_kernel, mode="valid"),
        0,
        padded_time,
    )

    lon_pad = len(lon_kernel) // 2
    padded_lon = np.pad(smoothed_time, ((0, 0), (lon_pad, lon_pad)), mode="wrap")
    return np.apply_along_axis(
        lambda row: np.convolve(row, lon_kernel, mode="valid"),
        1,
        padded_lon,
    )


def main() -> None:
    data = np.load(DATA_FILE)
    lon = np.asarray(data["longitude"], dtype=float)
    times = data["time"].astype("datetime64[m]").astype(object)
    anomaly = np.asarray(data["u_anomaly"], dtype=float)
    anomaly = smooth_hovmoller(anomaly)
    forecast_start = data["forecast_start"].astype("datetime64[m]").item()
    reference = str(data["reference_period"].item())
    is_anomaly = reference != "operational-raw-wind"

    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, cmap.N, clip=True)
    fig, ax = plt.subplots(figsize=(10, 11))
    mesh = ax.contourf(lon, times, anomaly, levels=LEVELS, cmap=cmap,
                       norm=norm, extend="both")

    # Positive (westerly) extreme contours highlight Kelvin-wave-like bursts.
    west = ax.contour(lon, times, anomaly, levels=[6, 9, 12],
                      colors="black", linewidths=[0.7, 1.0, 1.35])
    ax.clabel(west, fmt="+%d", fontsize=7, inline=True)
    ax.axhline(forecast_start, color="#e99aac", linewidth=5, alpha=0.8)
    ax.text(180, forecast_start, "  Begin GFS/GEFS Forecast",
            ha="center", va="top", fontsize=10, color="black")

    ax.set_xlim(0, 360)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(which="major", color="0.5", linewidth=0.35, alpha=0.35)
    title = ("850-hPa Zonal Wind Anomalies (1991–2020)" if is_anomaly
             else "850-hPa Zonal Wind — Operational")
    ax.set_title(title, fontsize=20, pad=12)
    ax.text(1.0, 1.015, "[5°S–5°N]", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=16)

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.09,
                        fraction=0.05, ticks=np.arange(-15, 16, 3))
    quantity = "Zonal wind anomaly" if is_anomaly else "Zonal wind"
    cbar.set_label(f"{quantity} [m s⁻¹]   Blue: easterly | Red: westerly", fontsize=11)
    fig.text(0.5, 0.015,
             "Black contours (+6, +9, +12 m s⁻¹) emphasize extreme westerly wind bursts"
             + ("" if is_anomaly else " | NOAA climatology temporarily unavailable"),
             ha="center", fontsize=9)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
