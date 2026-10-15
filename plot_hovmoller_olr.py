"""Plot operational tropical outgoing-longwave-radiation Hovmoller."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


DATA_FILE = Path("data/hovmoller_olr_raw.npz")
OUT_FILE = Path("output/HOVMOLLER_OLR_LATEST.png")
ANOMALY_FILE = Path("output/HOVMOLLER_OLR_30DAY_DEPARTURE_LATEST.png")
EASTWARD_FILE = Path("output/HOVMOLLER_OLR_EASTWARD_FILTERED_LATEST.png")
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


def eastward_filter(raw: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Retain eastward 3–15-day, zonal waves 1–10 in the hybrid series.

    This is a directional spectral diagnostic, not a Kelvin dispersion filter
    or a 20–100-day MJO filter. Temporal end effects remain near both edges.
    """
    stamps = np.asarray(times, dtype="datetime64[m]")
    if raw.shape[0] < 120 or raw.shape[1] < 20:
        raise ValueError("Eastward OLR filter needs at least 30 days of data")
    if not np.all(np.diff(stamps) == np.timedelta64(6, "h")):
        raise ValueError("Eastward OLR filter needs uninterrupted six-hourly data")
    if not np.isfinite(raw).all():
        raise ValueError("Eastward OLR filter needs finite OLR values")
    values = raw.astype(float) - raw.mean(axis=0, keepdims=True)
    x = np.arange(len(values), dtype=float) - (len(values) - 1) / 2
    slope = (x[:, None] * values).sum(axis=0) / np.dot(x, x)
    values -= x[:, None] * slope[None, :]
    # A short cosine taper avoids wrapping an abrupt time-series edge into
    # spurious propagating waves; the first/last four days are less reliable.
    edge = min(16, len(values) // 8)
    taper = np.ones(len(values))
    ramp = np.sin(np.linspace(0, np.pi / 2, edge, endpoint=False)) ** 2
    taper[:edge] = ramp
    taper[-edge:] = ramp[::-1]
    spectrum = np.fft.fft2(values * taper[:, None])
    frequency = np.fft.fftfreq(len(values), d=0.25)[:, None]
    wave = np.fft.fftfreq(values.shape[1], d=1 / values.shape[1])[None, :]
    keep = ((np.abs(frequency) >= 1 / 15) & (np.abs(frequency) <= 1 / 3)
            & (np.abs(wave) >= 1) & (np.abs(wave) <= 10)
            & (frequency * wave < 0))
    return np.fft.ifft2(spectrum * keep).real


def plot_eastward(lon, times, raw, forecast_start, cycle):
    filtered = eastward_filter(raw, times)
    levels = np.arange(-35, 36, 5)
    fig, ax = plt.subplots(figsize=(10, 11))
    mesh = ax.contourf(lon, times, filtered, levels=levels, cmap="BrBG_r",
                       extend="both")
    ax.axhline(forecast_start, color="#e99aac", linewidth=4, alpha=0.85)
    ax.text(180, forecast_start, "  Begin GFS Forecast", ha="center", va="top",
            fontsize=10)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(color="0.5", linewidth=0.35, alpha=0.35)
    ax.set_title("Eastward-filtered tropical OLR", fontsize=16, pad=12, loc="left")
    ax.text(1, 1.015, "[15°S–15°N]", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=13)
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.09,
                        fraction=0.05, ticks=np.arange(-30, 31, 10))
    cbar.set_label("Filtered OLR [W m⁻²] | Teal: enhanced convection", fontsize=10)
    fig.text(0.5, 0.015,
             f"Eastward 3–15 d, zonal waves 1–10 | GFS analysis + forecast | "
             f"End effects ~4 d; not MJO/Kelvin-specific | {cycle}",
             ha="center", fontsize=8)
    EASTWARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(EASTWARD_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {EASTWARD_FILE}")


def plot_departure(lon, times, raw, forecast_start, cycle):
    """Show departures from a fixed observed 30-day reference, not climatology.

    The reference stops before the forecast starts, preventing future forecast
    values from contaminating the baseline. This is a short-term diagnostic;
    it is not a 20–100-day MJO band-pass or a seasonal anomaly.
    """
    stamps = np.asarray(times, dtype="datetime64[m]")
    start = np.datetime64(forecast_start, "m")
    reference = (stamps < start) & (stamps >= start - np.timedelta64(30, "D"))
    if reference.sum() < 60:
        raise ValueError("OLR departure requires at least 15 days of six-hourly history")
    baseline = raw[reference].mean(axis=0)
    departure = smooth(raw - baseline[None, :])
    # A second, spatially smoothed signal makes broad tropical envelopes easier
    # to see. It does not claim to isolate a particular propagating wave.
    levels = np.arange(-60, 61, 5)
    fig, ax = plt.subplots(figsize=(10, 11))
    mesh = ax.contourf(lon, times, departure, levels=levels, cmap="BrBG_r",
                       extend="both")
    wet = ax.contour(lon, times, departure, levels=[-40, -25, -15],
                     colors="#115d71", linewidths=[1.2, 1, 0.8])
    ax.clabel(wet, fmt="%d", fontsize=7)
    ax.axhline(forecast_start, color="#e99aac", linewidth=4, alpha=0.85)
    ax.text(180, forecast_start, "  Begin GFS Forecast", ha="center", va="top",
            fontsize=10)
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W", "0°"])
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(color="0.5", linewidth=0.35, alpha=0.35)
    ax.set_title("OLR departure — previous 30-day mean", fontsize=16,
                 pad=12, loc="left")
    ax.text(1, 1.015, "[15°S–15°N]", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=13)
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.09,
                        fraction=0.05, ticks=np.arange(-60, 61, 15))
    cbar.set_label("OLR departure [W m⁻²] | Negative: enhanced clouds/convection", fontsize=10)
    fig.text(0.5, 0.015,
             f"Fixed observed 30-day reference; not climatology or a wave band-pass | GFS {cycle}",
             ha="center", fontsize=9)
    ANOMALY_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ANOMALY_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {ANOMALY_FILE}")


def main() -> None:
    data = np.load(DATA_FILE)
    lon = np.asarray(data["longitude"], dtype=float)
    times = data["time"].astype("datetime64[m]").astype(object)
    raw = np.asarray(data["olr"], dtype=float)
    olr = smooth(raw)
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
    plot_departure(lon, times, raw, start, cycle)
    plot_eastward(lon, times, raw, start, cycle)


if __name__ == "__main__":
    main()
