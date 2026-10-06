"""Hybrid GFS U850 forecast with an eastward equatorial Kelvin filter."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import requests

import u850_kelvin_scale as wind

DATA_DIR = Path("data/u850_kelvin_filtered")
OUT_FILE = Path("output/U850_KELVIN_FILTERED_FORECAST_LATEST.png")
HISTORY_DAYS = 9
MIN_HISTORY_DAYS = 7
FORECAST_DAYS = 14


def download_one(cycle, day):
    # Negative days use the GFS f000 analyses from their own valid cycles.
    valid = cycle + timedelta(days=day)
    params = wind.params(valid, 0) if day < 0 else wind.params(cycle, day * 24)
    path = DATA_DIR / f"u850_{day:+03d}.grib2"
    for attempt in range(4):
        try:
            response = requests.get(wind.kelvin.base.base.FILTER, params=params, timeout=180)
            if wind.kelvin.base.base.valid(response):
                path.write_bytes(response.content)
                return day, path
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return day, None


def download_series():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    days = range(-HISTORY_DAYS, FORECAST_DAYS + 1)
    for cycle in wind.kelvin.base.base.candidates():
        with ThreadPoolExecutor(max_workers=5) as pool:
            paths = dict(f.result() for f in as_completed(
                [pool.submit(download_one, cycle, day) for day in days]))
        history = 0
        while history < HISTORY_DAYS and paths.get(-history - 1) is not None:
            history += 1
        if history >= MIN_HISTORY_DAYS and all(paths.get(day) for day in range(FORECAST_DAYS + 1)):
            return cycle, history, paths
    raise RuntimeError("No complete GFS U850 daily forecast with 7 consecutive analysis days")


def build_anomalies(paths, history):
    fields, lat, lon = [], None, None
    for day in range(-history, FORECAST_DAYS + 1):
        u, this_lat, this_lon = wind.read_u(paths[day])
        if lat is not None and (not np.array_equal(lat, this_lat) or
                                not np.array_equal(lon, this_lon)):
            raise ValueError("GFS U850 grids differ across the daily series")
        lat, lon = this_lat, this_lon
        fields.append(u - np.nanmean(u, axis=1, keepdims=True))
    values = np.asarray(fields)
    if not np.isfinite(values).all():
        raise ValueError("Missing U850 values in daily series")
    return values, lat, lon


def plot(filtered, lat, lon, cycle, history):
    levels = np.arange(-5, 5.5, 0.5)
    norm = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11.5),
                            subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)})
    mesh = None
    for ax, day in zip(axes, (0, 3, 6)):
        field = filtered[history + day]
        mesh = ax.contourf(lon, lat, field, levels=levels, cmap="RdBu_r",
                           norm=norm, extend="both", transform=ccrs.PlateCarree())
        ax.contour(lon, lat, field, levels=(-3, -1.5, 1.5, 3), colors="black",
                   linewidths=0.3, alpha=0.5, transform=ccrs.PlateCarree())
        ax.coastlines(resolution="110m", linewidth=0.7)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25)
        ax.set_extent([0, 360, -25, 25], crs=ccrs.PlateCarree())
        label = "Today" if day == 0 else f"+{day} days"
        ax.set_title(f"{label}  |  {cycle + timedelta(days=day):%d %b %Y %H UTC}", loc="left")
        grid = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.3)
        grid.top_labels = grid.right_labels = False
    fig.suptitle("GFS Eastward Kelvin-filtered U850 Forecast", fontsize=18, y=0.99)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", pad=0.04,
                        fraction=0.04, ticks=np.arange(-5, 6, 1))
    cbar.set_label("Filtered U850 [m s⁻¹]   Red: westerly | Blue: easterly")
    fig.text(0.5, 0.012,
             f"Hybrid {history}-day GFS analyses + 14-day forecast | symmetric eastward 2.5–20 d, waves 1–14, h=8–90 m",
             ha="center", fontsize=9)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    cycle, history, paths = download_series()
    values, lat, lon = build_anomalies(paths, history)
    filtered = wind.kelvin.kelvin_filter(values, lat)
    plot(filtered, lat, lon, cycle, history)
    print(f"Saved {OUT_FILE} from GFS {cycle:%Y-%m-%d %H} UTC")


if __name__ == "__main__":
    main()
