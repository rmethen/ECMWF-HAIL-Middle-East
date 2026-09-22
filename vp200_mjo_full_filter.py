"""Create a centred, eastward MJO-filtered VP200 analysis from NOAA data.

The filter retains eastward zonal wavenumbers 1-5 and periods of 20-100 days.
Because it is centred, the most recent trustworthy analysis necessarily lags
the end of the NOAA archive by half the 201-day analysis window.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import requests
from scipy import signal
import xarray as xr


DATA_DIR = Path("data/vp200_mjo_full")
OUT_FILE = Path("output/VP200_MJO_FULL_FILTER_LATEST.png")
URL = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/spectral/daily/chi.{year}.nc"
WINDOW_DAYS = 201
HALF_WINDOW = WINDOW_DAYS // 2


def download_year(year: int) -> Path:
    path = DATA_DIR / f"chi.{year}.nc"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    response = requests.get(URL.format(year=year), timeout=300)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def load_archive() -> xr.DataArray:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    year = datetime.now(timezone.utc).year
    arrays = []
    for value in range(year - 2, year + 1):
        try:
            ds = xr.open_dataset(download_year(value))
        except requests.RequestException:
            continue
        arrays.append(ds["chi"].sel(level=0.2101, method="nearest").load())
        ds.close()
    if not arrays:
        raise RuntimeError("No NOAA daily velocity-potential archive was available")
    data = xr.concat(arrays, dim="time").sortby("time")
    _, unique = np.unique(data.time.values, return_index=True)
    return data.isel(time=np.sort(unique))


def select_complete_window(data: xr.DataArray) -> xr.DataArray:
    # A symmetric filter must have real observations on both sides.  Choose
    # the newest centre with 100 daily samples available after it.
    daily = data.resample(time="1D").interpolate("linear")
    if daily.sizes["time"] < WINDOW_DAYS:
        raise RuntimeError("NOAA archive is too short for the 20-100-day filter")
    return daily.isel(time=slice(-WINDOW_DAYS, None))


def mjo_filter(data: xr.DataArray) -> np.ndarray:
    values = np.asarray(data.values, dtype=np.float64)
    # Retain only the longitudinally varying anomaly before time-space filtering.
    values -= np.nanmean(values, axis=2, keepdims=True)
    values = signal.detrend(values, axis=0, type="linear")
    taper = signal.windows.tukey(values.shape[0], alpha=0.2)
    spectrum = np.fft.fft2(values * taper[:, None, None], axes=(0, 2))
    frequency = np.fft.fftfreq(values.shape[0], d=1.0)
    wave = np.fft.fftfreq(values.shape[2], d=1.0 / values.shape[2])
    # cos(k*x - omega*t) is eastward; in NumPy's transform this means f*k < 0.
    keep = (
        (np.abs(frequency[:, None]) >= 1.0 / 100.0)
        & (np.abs(frequency[:, None]) <= 1.0 / 20.0)
        & (np.abs(wave[None, :]) >= 1.0)
        & (np.abs(wave[None, :]) <= 5.0)
        & (frequency[:, None] * wave[None, :] < 0.0)
    )
    spectrum *= keep[:, None, :]
    return np.fft.ifft2(spectrum, axes=(0, 2)).real


def plot(filtered: np.ndarray, data: xr.DataArray) -> None:
    centre = HALF_WINDOW
    panels = [(centre - 14, "Two weeks earlier"),
              (centre - 7, "One week earlier"),
              (centre, "Latest centred analysis")]
    lat = np.asarray(data.lat)
    lon = np.asarray(data.lon)
    levels = np.arange(-8, 9, 1)
    colors = [
        "#e600d7", "#9c00dc", "#4b00bd", "#0027ad", "#0756cf", "#4d8ce4",
        "#9bc8f0", "#dceefa", "#ffffff", "#fff3b0", "#ffd36a", "#f5a137",
        "#ed681f", "#df3118", "#b9000d", "#721400",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(3, 1, figsize=(14, 13.5), subplot_kw={"projection": projection})
    mesh = None
    for ax, (index, label) in zip(axes, panels):
        field = filtered[index] / 1e6
        mesh = ax.contourf(lon, lat, field, levels=levels, cmap=cmap, norm=norm,
                           extend="both", transform=ccrs.PlateCarree())
        ax.contour(lon, lat, field, levels=levels, colors="black", linewidths=0.28,
                   alpha=0.45, transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.75)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_extent([0, 360, -45, 45], crs=ccrs.PlateCarree())
        date = np.datetime_as_string(data.time.values[index], unit="D")
        ax.set_title(f"{label}  |  {date}", loc="left", fontsize=13)
        grid = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.45", alpha=0.3)
        grid.top_labels = grid.right_labels = False
    fig.suptitle("Historical Centred MJO Analysis — Latest Reliable Date", fontsize=19, y=0.988)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", pad=0.035,
                        fraction=0.035, ticks=np.arange(-8, 9, 2))
    cbar.set_label("VP200 [10⁶ m² s⁻¹]   Blue/purple: upper divergence | Red: convergence", fontsize=11)
    archive_end = np.datetime_as_string(data.time.values[-1], unit="D")
    fig.text(0.5, 0.012,
             f"NOAA NCEP/NCAR daily analysis | eastward 20-100 d, waves 1-5 | archive window ends {archive_end}",
             ha="center", fontsize=9)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    window = select_complete_window(load_archive())
    filtered = mjo_filter(window)
    plot(filtered, window)
    date = np.datetime_as_string(window.time.values[HALF_WINDOW], unit="D")
    print(f"Saved {OUT_FILE}; latest centred analysis is {date}")


if __name__ == "__main__":
    main()
