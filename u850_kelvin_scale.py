"""GFS equatorial Kelvin-filtered 850-hPa zonal wind forecast."""

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
import xarray as xr

import vp200_kelvin_scale as kelvin


DATA_DIR = Path("data/u850_kelvin_hybrid")
OUT_FILE = Path("output/U850_KELVIN_SCALE_FORECAST_LATEST.png")
HISTORY_DAYS = 9
MIN_HISTORY_DAYS = 7
LEADS = tuple(range(0, 337, 24))


def params(valid, lead):
    return {
        "file": f"gfs.t{valid:%H}z.pgrb2.1p00.f{lead:03d}",
        "lev_850_mb": "on", "var_UGRD": "on",
        "subregion": "", "leftlon": 0, "rightlon": 360,
        "toplat": 25, "bottomlat": -25,
        "dir": f"/gfs.{valid:%Y%m%d}/{valid:%H}/atmos",
    }


def download_one(cycle, lead=None, days_back=None):
    valid = cycle - timedelta(days=days_back) if days_back is not None else cycle
    offset = f"m{days_back:02d}" if days_back is not None else f"f{lead:03d}"
    path = DATA_DIR / f"gfs_u850_{offset}.grib2"
    for attempt in range(4):
        try:
            response = requests.get(kelvin.base.base.FILTER,
                                    params=params(valid, 0 if days_back is not None else lead),
                                    timeout=180)
            if kelvin.base.base.valid(response):
                path.write_bytes(response.content)
                return offset, path
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return offset, None


def download_series():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cycle in kelvin.base.base.candidates():
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(download_one, cycle, lead=lead) for lead in LEADS]
            forecasts = dict(future.result() for future in as_completed(futures))
        if any(forecasts[f"f{lead:03d}"] is None for lead in LEADS):
            continue
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(download_one, cycle, days_back=day)
                       for day in range(1, HISTORY_DAYS + 1)]
            history = dict(future.result() for future in as_completed(futures))
        count = 0
        for day in range(1, HISTORY_DAYS + 1):
            if history[f"m{day:02d}"] is None:
                break
            count = day
        if count < MIN_HISTORY_DAYS:
            continue
        records = [(cycle - timedelta(days=day), history[f"m{day:02d}"])
                   for day in range(count, 0, -1)]
        records.extend((cycle + timedelta(hours=lead), forecasts[f"f{lead:03d}"])
                       for lead in LEADS)
        return cycle, count, records
    raise RuntimeError("No GFS cycle has a complete forecast and seven consecutive analyses")


def read_u(path):
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "level": 850},
            "indexpath": ""}) as ds:
        name = "u" if "u" in ds else next(n for n in ds.data_vars if n.lower().startswith("u"))
        return (np.asarray(ds[name].squeeze().values, dtype=np.float64),
                np.asarray(ds.latitude, dtype=np.float64),
                np.asarray(ds.longitude, dtype=np.float64))


def build_fields(records):
    fields = []
    lat_out = lon_out = None
    for _, path in records:
        u, lat, lon = read_u(path)
        if lat_out is not None and (not np.array_equal(lat, lat_out)
                                    or not np.array_equal(lon, lon_out)):
            raise ValueError("GFS grids differ within the Kelvin series")
        # The eastward k=1..14 filter removes the zonal mean; no climatology
        # or raw-wind fallback is presented as an anomaly.
        fields.append(u - np.mean(u, axis=1, keepdims=True))
        lat_out, lon_out = lat, lon
    return np.asarray(fields), lat_out, lon_out


def plot(filtered, lat, lon, cycle, history_days):
    panels = [(history_days, "Today", 0), (history_days + 3, "+3 days", 3),
              (history_days + 6, "+6 days", 6)]
    levels = np.arange(-8, 9, 1)
    cmap = plt.get_cmap("RdBu_r", len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11.5),
                             subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)})
    mesh = None
    for ax, (index, label, lead) in zip(axes, panels):
        field = filtered[index]
        mesh = ax.contourf(lon, lat, field, levels=levels, cmap=cmap, norm=norm,
                           extend="both", transform=ccrs.PlateCarree())
        ax.contour(lon, lat, field, levels=levels, colors="black", linewidths=0.2,
                   alpha=0.35, transform=ccrs.PlateCarree())
        ax.coastlines(resolution="110m", linewidth=0.7)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25)
        ax.set_extent([0, 360, -25, 25], crs=ccrs.PlateCarree())
        ax.set_title(f"{label}  |  {cycle + timedelta(days=lead):%d %b %Y %H UTC}",
                     loc="left", fontsize=12)
        grid = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.45", alpha=0.3)
        grid.top_labels = grid.right_labels = False
    fig.suptitle("GFS Eastward Kelvin-filtered U850 Forecast", fontsize=18, y=0.99)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", pad=0.04,
                        fraction=0.04, ticks=np.arange(-8, 9, 2))
    cbar.set_label("Filtered zonal wind at 850 hPa [m s⁻¹]   Red: westerly | Blue: easterly")
    fig.text(0.5, 0.012,
             f"Hybrid {history_days}-day GFS analyses + 14-day forecast | symmetric eastward 2.5-20 d, waves 1-14, h=8-90 m",
             ha="center", fontsize=9)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    cycle, history_days, records = download_series()
    values, lat, lon = build_fields(records)
    plot(kelvin.kelvin_filter(values, lat), lat, lon, cycle, history_days)
    print(f"Saved {OUT_FILE} from GFS {cycle:%Y-%m-%d %H} UTC")


if __name__ == "__main__":
    main()
