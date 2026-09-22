"""Hybrid GFS analysis/forecast VP200 filtered for eastward Kelvin waves."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
import time
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
import requests
from scipy import signal
import vp200_mjo_scale as base

DATA_DIR = Path("data/vp200_kelvin_hybrid")
OUT_FILE = Path("output/VP200_KELVIN_SCALE_FORECAST_LATEST.png")
HISTORY_DAYS = 9
MIN_HISTORY_DAYS = 7

def download_analysis(cycle, days_back):
    valid = cycle - timedelta(days=days_back)
    path = DATA_DIR / f"gfs_uv200_m{days_back:02d}.grib2"
    for attempt in range(5):
        try:
            response = requests.get(base.base.FILTER, params=base.base.params(valid), timeout=180)
            if base.base.valid(response):
                path.write_bytes(response.content)
                return days_back, path
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return days_back, None

def download_series():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base.DATA_DIR = DATA_DIR
    cycle, forecast_paths = base.download_forecast()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_analysis, cycle, day) for day in range(1, HISTORY_DAYS + 1)]
        history_paths = dict(future.result() for future in as_completed(futures))
    history_days = 0
    for day in range(1, HISTORY_DAYS + 1):
        if history_paths.get(day) is None:
            break
        history_days = day
    if history_days < MIN_HISTORY_DAYS:
        raise RuntimeError(f"Only {history_days} consecutive GFS analysis days were available")
    records = [(cycle - timedelta(days=day), history_paths[day]) for day in range(history_days, 0, -1)]
    records.extend((cycle + timedelta(hours=lead), forecast_paths[lead]) for lead in base.LEADS)
    return cycle, history_days, records

def build_anomalies(records):
    fields, lat_out, lon_out = [], None, None
    for valid, path in records:
        u, v, lat, lon = base.read_wind(path)
        chi, chi_lat = base.base.velocity_potential(u, v, lat)
        chi -= np.nanmean(chi, axis=1, keepdims=True)
        fields.append(chi)
        lat_out, lon_out = chi_lat, lon
    return np.asarray(fields), lat_out, lon_out

def kelvin_filter(values, lat):
    """Eastward 2.5-20 d, waves 1-14, equivalent depths 8-90 m."""
    opposite = np.asarray([np.argmin(np.abs(lat + value)) for value in lat])
    values = 0.5 * (values + values[:, opposite, :])
    values = signal.detrend(values, axis=0, type="linear")
    taper = signal.windows.tukey(values.shape[0], alpha=0.15)
    ntime, nfft = values.shape[0], 64
    spectrum = np.fft.fft2(values * taper[:, None, None], s=(nfft, values.shape[2]), axes=(0, 2))
    frequency = np.fft.fftfreq(nfft, d=1.0)
    wave = np.fft.fftfreq(values.shape[2], d=1.0 / values.shape[2])
    circumference = 2.0 * np.pi * 6_371_000.0
    fmin = np.sqrt(9.80665 * 8.0) * np.abs(wave) * 86_400.0 / circumference
    fmax = np.sqrt(9.80665 * 90.0) * np.abs(wave) * 86_400.0 / circumference
    keep = ((np.abs(frequency[:, None]) >= 1.0 / 20.0)
            & (np.abs(frequency[:, None]) <= 1.0 / 2.5)
            & (np.abs(wave[None, :]) >= 1.0) & (np.abs(wave[None, :]) <= 14.0)
            & (np.abs(frequency[:, None]) >= fmin[None, :])
            & (np.abs(frequency[:, None]) <= fmax[None, :])
            & (frequency[:, None] * wave[None, :] < 0.0))
    spectrum *= keep[:, None, :]
    return np.fft.ifft2(spectrum, axes=(0, 2)).real[:ntime]

def plot(filtered, lat, lon, cycle, history_days):
    panels = [(history_days, "Today", 0), (history_days + 3, "+3 days", 3),
              (history_days + 6, "+6 days", 6)]
    levels = np.arange(-4, 4.5, 0.5)
    cmap = plt.get_cmap("RdYlBu_r", len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N)
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11.5), subplot_kw={"projection": projection})
    mesh = None
    for ax, (index, label, lead) in zip(axes, panels):
        field = filtered[index] / 1e6
        mesh = ax.contourf(lon, lat, field, levels=levels, cmap=cmap, norm=norm,
                           extend="both", transform=ccrs.PlateCarree())
        ax.contour(lon, lat, field, levels=levels, colors="black", linewidths=0.2,
                   alpha=0.35, transform=ccrs.PlateCarree())
        ax.coastlines(resolution="110m", linewidth=0.7)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25)
        ax.set_extent([0, 360, -25, 25], crs=ccrs.PlateCarree())
        ax.set_title(f"{label}  |  {cycle + timedelta(days=lead):%d %b %Y %H UTC}", loc="left", fontsize=12)
        grid = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.45", alpha=0.3)
        grid.top_labels = grid.right_labels = False
    fig.suptitle("GFS Eastward Kelvin-filtered VP200 Forecast", fontsize=18, y=0.99)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", pad=0.04,
                        fraction=0.04, ticks=np.arange(-4, 5, 1))
    cbar.set_label("Filtered VP200 [10⁶ m² s⁻¹]   Blue: upper divergence | Red: convergence")
    fig.text(0.5, 0.012,
             f"Hybrid {history_days}-day GFS analyses + 14-day forecast | symmetric eastward 2.5-20 d, waves 1-14, h=8-90 m",
             ha="center", fontsize=9)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main():
    cycle, history_days, records = download_series()
    values, lat, lon = build_anomalies(records)
    plot(kelvin_filter(values, lat), lat, lon, cycle, history_days)
    print(f"Saved {OUT_FILE} from GFS {cycle:%Y-%m-%d %H} UTC")

if __name__ == "__main__":
    main()
