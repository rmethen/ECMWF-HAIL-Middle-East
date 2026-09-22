"""Experimental GFS OLR map with internally derived tropical-wave guides."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import requests
from scipy import signal
import xarray as xr

import vp200_unfiltered as base


DATA_DIR = Path("data/gfs_tropical_waves")
OUT_FILE = Path("output/GFS_OLR_MJO_KELVIN_ER_LATEST.png")
HISTORY_DAYS = 7
FORECAST_DAYS = 8


def wind_params(cycle, lead):
    return {
        "file": f"gfs.t{cycle:%H}z.pgrb2.1p00.f{lead:03d}",
        "lev_200_mb": "on", "var_UGRD": "on", "var_VGRD": "on",
        "subregion": "", "leftlon": 0, "rightlon": 360,
        "toplat": 90, "bottomlat": -90,
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
    }


def olr_params(cycle, lead):
    return {
        "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}",
        "lev_top_of_atmosphere": "on", "var_ULWRF": "on",
        "subregion": "", "leftlon": 0, "rightlon": 360,
        "toplat": 35, "bottomlat": -35,
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
    }


def fetch(cycle, lead, name, olr=False):
    path = DATA_DIR / f"{name}.grib2"
    if path.exists() and path.stat().st_size > 10_000:
        return path
    url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl" if olr else base.FILTER
    request_params = olr_params(cycle, lead) if olr else wind_params(cycle, lead)
    for attempt in range(5):
        try:
            response = requests.get(url, params=request_params, timeout=180)
            if base.valid(response):
                path.write_bytes(response.content)
                return path
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def download_series():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cycle in base.candidates():
        current = fetch(cycle, 0, "f000")
        if current is None:
            continue
        jobs = [(cycle, day * 24, f"f{day * 24:03d}", day)
                for day in range(1, FORECAST_DAYS + 1)]
        jobs += [(cycle - timedelta(days=day), 0, f"m{day:02d}", -day)
                 for day in range(1, HISTORY_DAYS + 1)]
        paths = {0: current}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch, c, lead, name): index
                       for c, lead, name, index in jobs}
            for future in as_completed(futures):
                paths[futures[future]] = future.result()
        history = 0
        for day in range(1, HISTORY_DAYS + 1):
            if paths.get(-day) is None:
                break
            history = day
        if history >= 6 and all(paths.get(day) is not None
                                for day in range(FORECAST_DAYS + 1)):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(fetch, cycle, 6 + day * 24,
                                       f"olr{day:02d}", True): day
                           for day in range(FORECAST_DAYS)}
                olr_paths = {futures[future]: future.result()
                             for future in as_completed(futures)}
            if any(olr_paths.get(day) is None for day in range(FORECAST_DAYS)):
                continue
            records = [(cycle - timedelta(days=day), paths[-day])
                       for day in range(history, 0, -1)]
            records += [(cycle + timedelta(days=day), paths[day])
                        for day in range(FORECAST_DAYS + 1)]
            return cycle, history, records, olr_paths
    raise RuntimeError("No complete recent GFS tropical-wave series was available")


def read_wind(path):
    wind = xr.open_dataset(path, engine="cfgrib", backend_kwargs={
        "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "level": 200},
        "indexpath": "",
    })
    u = np.asarray(wind["u"].squeeze(), dtype=np.float64)
    v = np.asarray(wind["v"].squeeze(), dtype=np.float64)
    lat = np.asarray(wind.latitude, dtype=np.float64)
    lon = np.asarray(wind.longitude, dtype=np.float64)
    wind.close()
    return u, v, lat, lon


def read_olr(path):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    name = next(iter(ds.data_vars))
    field = np.asarray(ds[name].squeeze(), dtype=np.float64)
    lat = np.asarray(ds.latitude, dtype=np.float64)
    lon = np.asarray(ds.longitude, dtype=np.float64)
    ds.close()
    field -= np.nanmean(field, axis=1, keepdims=True)
    return field[::4, ::4], lat[::4], lon[::4]


def wave_filter(values, lat, period, waves, eastward, depths=None):
    opposite = np.asarray([np.argmin(np.abs(lat + value)) for value in lat])
    values = 0.5 * (values + values[:, opposite, :])
    values = signal.detrend(values, axis=0, type="linear")
    ntime, nfft = values.shape[0], 64
    taper = signal.windows.tukey(ntime, alpha=0.15)
    spectrum = np.fft.fft2(values * taper[:, None, None],
                           s=(nfft, values.shape[2]), axes=(0, 2))
    frequency = np.fft.fftfreq(nfft, d=1.0)
    wave = np.fft.fftfreq(values.shape[2], d=1.0 / values.shape[2])
    keep = ((np.abs(frequency[:, None]) >= 1.0 / period[1])
            & (np.abs(frequency[:, None]) <= 1.0 / period[0])
            & (np.abs(wave[None, :]) >= waves[0])
            & (np.abs(wave[None, :]) <= waves[1]))
    direction = frequency[:, None] * wave[None, :]
    keep &= direction < 0 if eastward else direction > 0
    if depths is not None:
        circumference = 2 * np.pi * 6_371_000.0
        fmin = np.sqrt(9.80665 * depths[0]) * np.abs(wave) * 86400 / circumference
        fmax = np.sqrt(9.80665 * depths[1]) * np.abs(wave) * 86400 / circumference
        keep &= ((np.abs(frequency[:, None]) >= fmin[None, :])
                 & (np.abs(frequency[:, None]) <= fmax[None, :]))
    spectrum *= keep[:, None, :]
    return np.fft.ifft2(spectrum, axes=(0, 2)).real[:ntime]


def build(records):
    chi, lat_out, lon_out = [], None, None
    for _, path in records:
        u, v, lat, lon = read_wind(path)
        potential, chi_lat = base.velocity_potential(u, v, lat)
        potential -= np.nanmean(potential, axis=1, keepdims=True)
        chi.append(potential)
        lat_out, lon_out = chi_lat, lon
    return np.asarray(chi), lat_out, lon_out


def build_olr(paths):
    fields, lat, lon = [], None, None
    for day in range(FORECAST_DAYS):
        field, lat, lon = read_olr(paths[day])
        fields.append(field)
    return np.asarray(fields), lat, lon


def planetary(field, max_wave=5):
    spectrum = np.fft.rfft(field, axis=1)
    out = np.zeros_like(spectrum)
    out[:, 1:max_wave + 1] = spectrum[:, 1:max_wave + 1]
    return np.fft.irfft(out, n=field.shape[1], axis=1)


def plot(cycle, history, olr, olr_lat, olr_lon, chi, kelvin, er, lat, lon):
    panels = [(history + day, day) for day in (0, 2, 4, 6)]
    levels = [-55, -40, -30, -20, -10, 10, 20, 30, 40, 55]
    colors = ["#07594f", "#328d81", "#72bab0", "#b8ddd7", "#f6f4e9",
              "#efe2bd", "#d6b779", "#ae7730", "#6f3f0c"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(2, 2, figsize=(15, 6.4),
                             subplot_kw={"projection": projection})
    mesh = None
    tropics = (lat >= -35) & (lat <= 35)
    for ax, (index, day) in zip(axes.flat, panels):
        end = min(index + 2, chi.shape[0])
        shade = np.mean(olr[day:day + 2], axis=0)
        mjo = planetary(np.mean(chi[index:end], axis=0)) / 1e6
        kw = np.mean(kelvin[index:end], axis=0) / 1e6
        rw = np.mean(er[index:end], axis=0) / 1e6
        mesh = ax.contourf(olr_lon, olr_lat, shade, levels=levels,
                           cmap=cmap, norm=norm, extend="both",
                           transform=ccrs.PlateCarree())
        ax.contour(lon, lat[tropics], mjo[tropics], levels=[-4, -2.5],
                   colors="black", linewidths=[1.8, 1.2], linestyles="solid",
                   transform=ccrs.PlateCarree())
        ax.contour(lon, lat[tropics], kw[tropics], levels=[-2, -1],
                   colors="#173cff", linewidths=[1.5, 1.0], linestyles="solid",
                   transform=ccrs.PlateCarree())
        ax.contour(lon, lat[tropics], rw[tropics], levels=[-2, -1],
                   colors="#f02a22", linewidths=[1.5, 1.0], linestyles="solid",
                   transform=ccrs.PlateCarree())
        ax.coastlines(resolution="110m", linewidth=0.65)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.2)
        ax.set_extent([0, 360, -35, 35], crs=ccrs.PlateCarree())
        start = cycle + timedelta(days=day)
        ax.set_title(f"{start:%d %b} to {start + timedelta(days=1):%d %b}",
                     loc="left", fontsize=12)
        grid = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.3)
        grid.top_labels = grid.right_labels = False
    fig.suptitle("Experimental GFS 2-day OLR with Tropical-Wave Guides", fontsize=18)
    fig.subplots_adjust(top=0.88, bottom=0.20, hspace=0.20, wspace=0.10)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", pad=0.07,
                        fraction=0.055)
    cbar.set_label("OLR departure from zonal mean [W m⁻²]   Teal: enhanced convection | Brown: suppressed")
    fig.text(0.5, 0.018,
             "Black: MJO-scale VP200 | Blue: eastward Kelvin | Red: westward ER (provisional short-window guide)",
             ha="center", fontsize=10)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    cycle, history, records, olr_paths = download_series()
    chi, lat, lon = build(records)
    olr, olr_lat, olr_lon = build_olr(olr_paths)
    kelvin = wave_filter(chi, lat, (2.5, 20), (1, 14), True, (8, 90))
    er = wave_filter(chi, lat, (9, 72), (1, 10), False, (8, 90))
    plot(cycle, history, olr, olr_lat, olr_lon, chi, kelvin, er, lat, lon)
    print(f"Saved {OUT_FILE} from GFS {cycle:%Y-%m-%d %H} UTC")


if __name__ == "__main__":
    main()
