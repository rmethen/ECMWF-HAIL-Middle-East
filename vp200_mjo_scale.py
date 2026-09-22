"""Build a three-panel GFS VP200 forecast at MJO planetary scales.

This product deliberately uses the name *MJO-scale* rather than claiming a
full Wheeler-Kiladis time-frequency filter.  It retains zonal wavenumbers 1-5
and averages the forecast into week-1 and week-2 windows.  The result is a
clean operational guide to the broad upper-level divergence envelope while
the longer historical archive needed for a 20-100-day filter is assembled.
"""

from __future__ import annotations

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
import xarray as xr

import vp200_unfiltered as base


DATA_DIR = Path("data/vp200_mjo_scale")
OUT_FILE = Path("output/VP200_MJO_SCALE_FORECAST_LATEST.png")
LEADS = tuple(range(0, 337, 24))


def params(cycle, lead):
    values = base.params(cycle)
    values["file"] = f"gfs.t{cycle:%H}z.pgrb2.1p00.f{lead:03d}"
    return values


def download_one(cycle, lead):
    path = DATA_DIR / f"gfs_uv200_f{lead:03d}.grib2"
    for attempt in range(4):
        try:
            response = requests.get(base.FILTER, params=params(cycle, lead), timeout=180)
            if base.valid(response):
                path.write_bytes(response.content)
                return lead, path
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GFS f{lead:03d} was unavailable")


def download_forecast():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cycle in base.candidates():
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(download_one, cycle, lead) for lead in LEADS]
                paths = dict(future.result() for future in as_completed(futures))
            return cycle, paths
        except RuntimeError:
            continue
    raise RuntimeError("No complete recent GFS VP200 forecast was available")


def read_wind(path):
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "level": 200},
            "indexpath": "",
        },
    )
    u_name = "u" if "u" in ds else next(n for n in ds.data_vars if n.lower().startswith("u"))
    v_name = "v" if "v" in ds else next(n for n in ds.data_vars if n.lower().startswith("v"))
    return (
        np.asarray(ds[u_name].squeeze().values, dtype=np.float64),
        np.asarray(ds[v_name].squeeze().values, dtype=np.float64),
        np.asarray(ds.latitude, dtype=np.float64),
        np.asarray(ds.longitude, dtype=np.float64),
    )


def planetary_filter(field, min_wave=1, max_wave=5):
    """Retain only the broad east-west planetary scales used by the MJO."""
    spectrum = np.fft.rfft(field, axis=1)
    filtered = np.zeros_like(spectrum)
    filtered[:, min_wave:max_wave + 1] = spectrum[:, min_wave:max_wave + 1]
    return np.fft.irfft(filtered, n=field.shape[1], axis=1)


def build_fields(cycle, paths):
    base.download_climatology()
    fields = {}
    lat_out = lon_out = None
    for lead in LEADS:
        u, v, lat, lon = read_wind(paths[lead])
        chi, chi_lat = base.velocity_potential(u, v, lat)
        valid = cycle + timedelta(hours=lead)
        anomaly = chi - base.climatology_for(valid, chi_lat, lon)
        anomaly -= np.nanmean(anomaly, axis=1, keepdims=True)
        fields[lead] = planetary_filter(anomaly)
        lat_out, lon_out = chi_lat, lon
    return fields, lat_out, lon_out


def plot(fields, lat, lon, cycle):
    panels = [
        (fields[0], "Today", cycle),
        (np.mean([fields[h] for h in range(24, 169, 24)], axis=0),
         "Week 1 mean", cycle + timedelta(days=4)),
        (np.mean([fields[h] for h in range(192, 337, 24)], axis=0),
         "Week 2 mean", cycle + timedelta(days=11)),
    ]
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
    for ax, (field, label, valid) in zip(axes, panels):
        mesh = ax.contourf(
            lon, lat, field / 1e6, levels=levels, cmap=cmap, norm=norm,
            extend="both", transform=ccrs.PlateCarree(),
        )
        ax.contour(
            lon, lat, field / 1e6, levels=levels, colors="black",
            linewidths=0.28, alpha=0.45, transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=0.75)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_extent([0, 360, -45, 45], crs=ccrs.PlateCarree())
        ax.set_title(f"{label}  |  centred {valid:%d %b %Y}", loc="left", fontsize=13)
        grid = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.45", alpha=0.3)
        grid.top_labels = grid.right_labels = False
    fig.suptitle("MJO-scale VP200 Forecast (Zonal Wavenumbers 1-5)", fontsize=19, y=0.988)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", pad=0.035,
                        fraction=0.035, ticks=np.arange(-8, 9, 2))
    cbar.set_label(
        "VP200 anomaly [10⁶ m² s⁻¹]   Blue/purple: upper divergence | Red: convergence",
        fontsize=11,
    )
    fig.text(
        0.5, 0.012,
        "GFS | 1991-2020 NCEP/NCAR climatology | planetary-scale spatial filter; not a full 20-100-day filter",
        ha="center", fontsize=9,
    )
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    cycle, paths = download_forecast()
    fields, lat, lon = build_fields(cycle, paths)
    plot(fields, lat, lon, cycle)
    print(f"Saved {OUT_FILE} from GFS {cycle:%Y-%m-%d %H} UTC")


if __name__ == "__main__":
    main()
