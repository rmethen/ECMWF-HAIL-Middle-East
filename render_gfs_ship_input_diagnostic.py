"""Render a REAL GFS precursor diagnostic, NOT a SHIP or hail forecast.

Uses actual downloaded pressure-level GRIB fields. SHIP remains blocked
until MU CAPE, MU mixing ratio, freezing AGL and 0–6km AGL shear are derived.
"""
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

SOURCE = Path("data/gfs_ship_f006.grib2")
DEST = Path("output/GFS_SHIP_INPUT_DIAGNOSTIC_F006.png")


def get(name):
    with xr.open_dataset(SOURCE, engine="cfgrib", backend_kwargs={
        "filter_by_keys": {"shortName": name, "typeOfLevel": "isobaricInhPa"},
        "indexpath": ""}) as ds:
        return ds[name].load()


def render():
    t, gh, u, v = (get(name) for name in ("t", "gh", "u", "v"))
    getlevel = lambda da, p: da.sel(isobaricInhPa=p).values
    lapse = ((getlevel(t, 700) - getlevel(t, 500)) /
             ((getlevel(gh, 500) - getlevel(gh, 700)) / 1000.0))
    shear_proxy = np.hypot(getlevel(u, 500) - getlevel(u, 850),
                           getlevel(v, 500) - getlevel(v, 850))
    lon, lat = t.longitude.values, t.latitude.values
    if not (np.isfinite(lapse).any() and np.isfinite(shear_proxy).any()):
        raise ValueError("No finite GFS profile data")
    if not (-20 < np.nanmedian(lapse) < 20):
        raise ValueError("Implausible lapse rate; inspect units")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, data, title, label, levels in [
        (axes[0], lapse, "700–500 hPa lapse rate", "K/km", np.arange(2, 11, .5)),
        (axes[1], shear_proxy, "850–500 hPa vector wind difference", "m/s", np.arange(0, 61, 3)),
    ]:
        im = ax.contourf(lon, lat, data, levels=levels, extend="both", cmap="viridis")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("Longitude E")
        ax.set_ylabel("Latitude N")
        ax.set_title(title)
        ax.grid(alpha=.2)
    valid = np.datetime_as_string(t.valid_time.values, unit="h")
    fig.suptitle("Real GFS +6h input diagnostics — NOT SHIP, NOT observed hail\n"
                 f"Valid UTC {valid}; SHIP MU parcel and AGL inputs still missing")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST, dpi=150)
    plt.close(fig)
    print(f"Created {DEST} ({DEST.stat().st_size} bytes)")


if __name__ == "__main__":
    render()
