"""GFS 850-hPa zonal-wind anomalies relative to each latitude's zonal mean.

This displays the full wind signal. It does not identify Kelvin waves.
"""

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
import xarray as xr

import vp200_kelvin_scale as kelvin

DATA_DIR = Path("data/u850_wind")
OUT_FILE = Path("output/U850_ZONAL_ANOMALY_FORECAST_LATEST.png")
LEADS = (0, 72, 144)


def params(cycle, lead):
    return {
        "file": f"gfs.t{cycle:%H}z.pgrb2.1p00.f{lead:03d}",
        "lev_850_mb": "on", "var_UGRD": "on",
        "subregion": "", "leftlon": 0, "rightlon": 360,
        "toplat": 25, "bottomlat": -25,
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
    }


def download_one(cycle, lead):
    path = DATA_DIR / f"gfs_u850_f{lead:03d}.grib2"
    for attempt in range(4):
        try:
            response = requests.get(kelvin.base.base.FILTER,
                                    params=params(cycle, lead), timeout=180)
            if kelvin.base.base.valid(response):
                path.write_bytes(response.content)
                return lead, path
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return lead, None


def download_series():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cycle in kelvin.base.base.candidates():
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(download_one, cycle, lead) for lead in LEADS]
            paths = dict(future.result() for future in as_completed(futures))
        if all(paths[lead] is not None for lead in LEADS):
            return cycle, paths
    raise RuntimeError("No GFS cycle has all three 850-hPa wind forecasts")


def read_u(path):
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "level": 850},
            "indexpath": ""}) as ds:
        name = "u" if "u" in ds else next(n for n in ds.data_vars if n.lower().startswith("u"))
        return (np.asarray(ds[name].squeeze().values, dtype=np.float64),
                np.asarray(ds.latitude, dtype=np.float64),
                np.asarray(ds.longitude, dtype=np.float64))


def plot(cycle, paths):
    records = [read_u(paths[lead]) for lead in LEADS]
    lat, lon = records[0][1:]
    for _, other_lat, other_lon in records[1:]:
        if not np.array_equal(lat, other_lat) or not np.array_equal(lon, other_lon):
            raise ValueError("GFS grids differ between forecast panels")
    fields = [u - np.nanmean(u, axis=1, keepdims=True) for u, _, _ in records]
    # A shared physical scale retains comparability between all three panels.
    levels = np.arange(-12, 14, 2)
    norm = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12)
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 11.5),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)})
    mesh = None
    for ax, lead, field in zip(axes, LEADS, fields):
        mesh = ax.contourf(lon, lat, field, levels=levels, cmap="RdBu_r",
                           norm=norm, extend="both", transform=ccrs.PlateCarree())
        ax.contour(lon, lat, field, levels=(-8, -4, 4, 8), colors="black",
                   linewidths=0.35, alpha=0.6, transform=ccrs.PlateCarree())
        ax.coastlines(resolution="110m", linewidth=0.7)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.25)
        ax.set_extent([0, 360, -25, 25], crs=ccrs.PlateCarree())
        label = "Today" if lead == 0 else f"+{lead // 24} days"
        ax.set_title(f"{label}  |  {cycle + timedelta(hours=lead):%d %b %Y %H UTC}",
                     loc="left", fontsize=12)
        grid = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.45", alpha=0.3)
        grid.top_labels = grid.right_labels = False
    fig.suptitle("GFS U850 Zonal Wind Anomaly Forecast", fontsize=18, y=0.99)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", pad=0.04,
                        fraction=0.04, ticks=np.arange(-12, 13, 4))
    cbar.set_label("U850 minus latitude zonal mean [m s⁻¹]   Red: westerly anomaly | Blue: easterly anomaly")
    fig.text(0.5, 0.012,
             "Unfiltered GFS wind anomaly | 25°S–25°N | This map alone does not identify a Kelvin wave",
             ha="center", fontsize=9)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    cycle, paths = download_series()
    plot(cycle, paths)
    print(f"Saved {OUT_FILE} from GFS {cycle:%Y-%m-%d %H} UTC")


if __name__ == "__main__":
    main()
