"""Build an unfiltered 200-hPa velocity-potential anomaly from operational GFS."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pyshtools
import requests
import xarray as xr


FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl"
CLIM_URL = (
    "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/"
    "spectral/chi.mon.ltm.1991-2020.nc"
)
DATA_DIR = Path("data/vp200")
GRIB_FILE = DATA_DIR / "gfs_uv200_global.grib2"
CLIM_FILE = DATA_DIR / "chi.mon.ltm.1991-2020.nc"
OUT_FILE = Path("output/VP200_UNFILTERED_ANOMALY_LATEST.png")
EARTH_RADIUS = 6_371_000.0


def candidates():
    now = datetime.now(timezone.utc)
    anchor = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0,
                         microsecond=0)
    for lag in range(0, 37, 6):
        yield anchor - timedelta(hours=lag)


def params(cycle):
    return {
        "file": f"gfs.t{cycle:%H}z.pgrb2.1p00.f000",
        "lev_200_mb": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "subregion": "",
        "leftlon": 0,
        "rightlon": 360,
        "toplat": 90,
        "bottomlat": -90,
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
    }


def valid(response):
    return response.ok and len(response.content) > 10_000 and response.content[:4] == b"GRIB"


def download_gfs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for cycle in candidates():
        for attempt in range(3):
            try:
                response = requests.get(FILTER, params=params(cycle), timeout=120)
                if valid(response):
                    GRIB_FILE.write_bytes(response.content)
                    return cycle
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
    raise RuntimeError("No recent GFS U/V 200-hPa global analysis was available")


def download_climatology():
    if CLIM_FILE.exists() and CLIM_FILE.stat().st_size > 1_000_000:
        return
    response = requests.get(CLIM_URL, timeout=240)
    response.raise_for_status()
    CLIM_FILE.write_bytes(response.content)


def read_wind():
    ds = xr.open_dataset(
        GRIB_FILE, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa",
                                             "level": 200},
                        "indexpath": ""},
    )
    u_name = "u" if "u" in ds else next(n for n in ds.data_vars if n.lower().startswith("u"))
    v_name = "v" if "v" in ds else next(n for n in ds.data_vars if n.lower().startswith("v"))
    u = np.asarray(ds[u_name].squeeze().values, dtype=np.float64)
    v = np.asarray(ds[v_name].squeeze().values, dtype=np.float64)
    lat = np.asarray(ds.latitude, dtype=np.float64)
    lon = np.asarray(ds.longitude, dtype=np.float64)
    return u, v, lat, lon


def velocity_potential(u, v, lat):
    # GFS 1-degree data include both poles (181 x 360).  Driscoll-Healy
    # sampling uses the north pole through 89S (180 x 360), so omit 90S.
    u = u[:-1]
    v = v[:-1]
    phi = np.deg2rad(lat[:-1])
    dlon = np.deg2rad(1.0)
    cosphi = np.cos(phi)
    du_dlambda = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dlon)
    dvcos_dphi = np.gradient(v * cosphi[:, None], phi, axis=0)
    div = (du_dlambda + dvcos_dphi) / (EARTH_RADIUS * cosphi[:, None])

    # Finite differences are singular at the poles.  Stabilise the outer
    # latitude rows and smoothly taper only poleward of 70 degrees; the
    # plotted tropical/subtropical solution is then not contaminated by a
    # spurious hemispheric dipole.
    div[:3] = div[3]
    div[-3:] = div[-4]
    abs_lat = np.abs(lat[:-1])
    taper = np.ones_like(abs_lat)
    polar = abs_lat > 70.0
    taper[polar] = 0.5 * (
        1.0 + np.cos(np.pi * (abs_lat[polar] - 70.0) / 20.0)
    )
    div *= taper[:, None]

    # A spherical divergence field must integrate to zero globally.  Tiny
    # numerical imbalances otherwise explode when the inverse Laplacian is
    # applied, especially in the lowest spherical-harmonic degrees.
    weights = cosphi[:, None]
    div -= np.sum(div * weights) / (np.sum(weights) * div.shape[1])

    coeffs = pyshtools.expand.SHExpandDH(div, sampling=2, norm=4)
    lmax = coeffs.shape[1] - 1
    # The NCEP/NCAR spectral climatology is T42.  Matching that effective
    # resolution removes grid-scale noise and makes the anomaly comparable.
    if lmax > 42:
        coeffs[:, 43:, :] = 0.0
    for degree in range(1, lmax + 1):
        coeffs[:, degree, :degree + 1] *= (
            -EARTH_RADIUS ** 2 / (degree * (degree + 1))
        )
    coeffs[:, 0, 0] = 0.0
    chi = pyshtools.expand.MakeGridDH(coeffs, sampling=2, norm=4)
    return chi, lat[:-1]


def climatology_for(cycle, target_lat, target_lon):
    ds = xr.open_dataset(CLIM_FILE)
    da = ds["chi"]
    level_name = "level"
    # NOAA provides sigma levels; 0.2101 is the level corresponding most
    # closely to 200 hPa.  The minimum (0.1682) is nearer 150–170 hPa and
    # creates a false hemispheric anomaly when compared with GFS 200 hPa.
    da = da.sel({level_name: 0.2101}, method="nearest")
    time_name = "time" if "time" in da.dims else "month"
    if time_name == "time":
        da = da.isel(time=cycle.month - 1)
    else:
        da = da.sel(month=cycle.month)
    da = da.sortby("lat").interp(
        lat=np.sort(target_lat), lon=target_lon, kwargs={"fill_value": "extrapolate"}
    ).sortby("lat", ascending=False)
    return np.asarray(da.values, dtype=np.float64)


def plot(anomaly, lat, lon, cycle):
    levels = np.arange(-10, 11, 1)
    colors = [
        "#ed00df", "#bd00e8", "#7a00d8", "#3600bd", "#0016a5",
        "#003ac5", "#1d65dd", "#5795eb", "#9bc5f2", "#d8eafa",
        "#ffffff", "#fff8c9", "#ffe98a", "#ffc953", "#f5a32d",
        "#ed741c", "#e64316", "#d51d12", "#b6000b", "#7c1400",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)
    projection = ccrs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=(14, 6.8))
    ax = plt.axes(projection=projection)
    mesh = ax.contourf(lon, lat, anomaly / 1e6, levels=levels, cmap=cmap,
                       norm=norm, extend="both", transform=ccrs.PlateCarree())
    ax.contour(lon, lat, anomaly / 1e6, levels=levels,
               colors="black", linewidths=0.32, alpha=0.55,
               transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.35)
    ax.set_extent([0, 360, -60, 60], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="0.45",
                      alpha=0.35, linestyle="--")
    gl.top_labels = gl.right_labels = False
    ax.set_title(f"{cycle:%d %b %Y %H UTC}", loc="left", fontsize=14)
    ax.set_title("Unfiltered VP200 Anomaly (Zonal Mean Removed)",
                 loc="right", fontsize=17)
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.08,
                        fraction=0.06, ticks=np.arange(-10, 11, 2))
    cbar.set_label(
        "Velocity potential anomaly [10⁶ m² s⁻¹]   Blue/purple: upper divergence | Red: convergence",
        fontsize=11,
    )
    fig.text(
        0.5, 0.018,
        "GFS 200-hPa wind | 1991–2020 NCEP/NCAR velocity-potential climatology",
        ha="center", fontsize=9,
    )
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    cycle = download_gfs()
    download_climatology()
    u, v, lat, lon = read_wind()
    chi, chi_lat = velocity_potential(u, v, lat)
    clim = climatology_for(cycle, chi_lat, lon)
    anomaly = chi - clim
    # Intraseasonal diagnostics (MJO/Kelvin) need the longitudinally varying
    # component.  Remove the latitude-by-latitude zonal mean so that changes
    # in the background Hadley circulation do not saturate the map and hide
    # eastward/westward propagating tropical signals.
    anomaly -= np.nanmean(anomaly, axis=1, keepdims=True)
    # Velocity potential is defined up to a constant; remove the area-weighted
    # global mean after differencing.
    weights = np.cos(np.deg2rad(chi_lat))[:, None]
    anomaly -= np.sum(anomaly * weights) / (np.sum(weights) * anomaly.shape[1])
    print(
        "VP200 ranges [10^6 m2 s-1] | "
        f"GFS {np.nanmin(chi) / 1e6:.1f}..{np.nanmax(chi) / 1e6:.1f} | "
        f"climatology {np.nanmin(clim) / 1e6:.1f}..{np.nanmax(clim) / 1e6:.1f} | "
        f"anomaly {np.nanmin(anomaly) / 1e6:.1f}..{np.nanmax(anomaly) / 1e6:.1f}"
    )
    plot(anomaly, chi_lat, lon, cycle)
    print(f"Saved {OUT_FILE} from GFS {cycle:%Y-%m-%d %H} UTC")


if __name__ == "__main__":
    main()
