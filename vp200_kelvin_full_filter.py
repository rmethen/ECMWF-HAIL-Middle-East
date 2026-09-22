"""Centred eastward Kelvin-filtered VP200 analysis from NOAA daily data."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from scipy import signal

import vp200_mjo_full_filter as base


OUT_FILE = Path("output/VP200_KELVIN_FULL_FILTER_LATEST.png")
WINDOW_DAYS = 61
HALF_WINDOW = 30


def select_complete_window(data):
    daily = data.resample(time="1D").interpolate("linear")
    if daily.sizes["time"] < WINDOW_DAYS:
        raise RuntimeError("NOAA archive is too short for the Kelvin filter")
    return daily.isel(time=slice(-WINDOW_DAYS, None))


def kelvin_filter(data):
    values = np.asarray(data.values, dtype=np.float64)
    values -= np.nanmean(values, axis=2, keepdims=True)
    values = 0.5 * (values + np.flip(values, axis=1))
    values = signal.detrend(values, axis=0, type="linear")
    taper = signal.windows.tukey(values.shape[0], alpha=0.25)
    spectrum = np.fft.fft2(values * taper[:, None, None], axes=(0, 2))
    frequency = np.fft.fftfreq(values.shape[0], d=1.0)
    wave = np.fft.fftfreq(values.shape[2], d=1.0 / values.shape[2])
    # Wheeler-Kiladis Kelvin dispersion envelope.  Equivalent depths 8-90 m
    # correspond to shallow-water phase speeds sqrt(g*h); retaining that
    # envelope rejects eastward features that are too slow or too fast to be
    # equatorial Kelvin waves.
    circumference = 2.0 * np.pi * 6_371_000.0
    speed_min = np.sqrt(9.80665 * 8.0)
    speed_max = np.sqrt(9.80665 * 90.0)
    dispersion_min = speed_min * np.abs(wave) * 86_400.0 / circumference
    dispersion_max = speed_max * np.abs(wave) * 86_400.0 / circumference
    keep = ((np.abs(frequency[:, None]) >= 1.0 / 20.0)
            & (np.abs(frequency[:, None]) <= 1.0 / 2.5)
            & (np.abs(wave[None, :]) >= 1.0)
            & (np.abs(wave[None, :]) <= 14.0)
            & (np.abs(frequency[:, None]) >= dispersion_min[None, :])
            & (np.abs(frequency[:, None]) <= dispersion_max[None, :])
            & (frequency[:, None] * wave[None, :] < 0.0))
    spectrum *= keep[:, None, :]
    return np.fft.ifft2(spectrum, axes=(0, 2)).real


def plot(filtered, data):
    panels = [(HALF_WINDOW - 6, "Six days earlier"),
              (HALF_WINDOW - 3, "Three days earlier"),
              (HALF_WINDOW, "Latest centred analysis")]
    lat, lon = np.asarray(data.lat), np.asarray(data.lon)
    levels = np.arange(-5, 5.5, 0.5)
    colors = plt.get_cmap("RdYlBu_r", len(levels) - 1)
    norm = BoundaryNorm(levels, colors.N)
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11.5), subplot_kw={"projection": projection})
    mesh = None
    for ax, (index, label) in zip(axes, panels):
        field = filtered[index] / 1e6
        mesh = ax.contourf(lon, lat, field, levels=levels, cmap=colors, norm=norm,
                           extend="both", transform=ccrs.PlateCarree())
        ax.contour(lon, lat, field, levels=levels, colors="black", linewidths=0.2,
                   alpha=0.35, transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)
        ax.set_extent([0, 360, -30, 30], crs=ccrs.PlateCarree())
        date = np.datetime_as_string(data.time.values[index], unit="D")
        ax.set_title(f"{label}  |  {date}", loc="left", fontsize=12)
        grid = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.45", alpha=0.3)
        grid.top_labels = grid.right_labels = False
    fig.suptitle("VP200 Kelvin Full Time-Space Filter", fontsize=18, y=0.99)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", pad=0.04,
                        fraction=0.04, ticks=np.arange(-5, 6, 1))
    cbar.set_label("VP200 [10⁶ m² s⁻¹]   Blue: upper divergence | Red: convergence")
    end = np.datetime_as_string(data.time.values[-1], unit="D")
    fig.text(0.5, 0.012,
             f"NOAA NCEP/NCAR | symmetric eastward 2.5-20 d, waves 1-14 | archive window ends {end}",
             ha="center", fontsize=9)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    window = select_complete_window(base.load_archive())
    filtered = kelvin_filter(window)
    plot(filtered, window)
    date = np.datetime_as_string(window.time.values[HALF_WINDOW], unit="D")
    print(f"Saved {OUT_FILE}; latest centred analysis is {date}")


if __name__ == "__main__":
    main()
