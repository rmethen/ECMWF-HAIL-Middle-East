"""Independent GFS large-hail environment map using the ECMWF V2 ingredients."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from hail_index import hail_potential_v2
from process_hail_grib import (approximate_li850, approximate_wet_bulb_c,
                               height_at_isotherm)


BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
LEVELS = (925, 850, 700, 600, 500, 400, 300, 200)
HOURS = range(6, 73, 6)
OUT = Path("output/GFS_HAIL_INDEX_MIDDLE_EAST_LATEST.png")
GRIB = Path("data/gfs_hail_latest.grib2")


def url(cycle, hour):
    params = {
        "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}",
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
        "subregion": "", "leftlon": 20, "rightlon": 65,
        "bottomlat": 10, "toplat": 45,
    }
    for name in ("TMP", "UGRD", "VGRD", "SPFH", "HGT", "VVEL"):
        params[f"var_{name}"] = "on"
    for level in LEVELS:
        params[f"lev_{level}_mb"] = "on"
    return BASE + "?" + urlencode(params)


def download(cycle, hour):
    with urlopen(Request(url(cycle, hour),
                         headers={"User-Agent": "ECMWF-HAIL-Middle-East/1.0"}),
                 timeout=60) as response:
        content = response.read()
    if not content.startswith(b"GRIB"):
        raise ValueError("NOMADS response did not contain GRIB")
    GRIB.parent.mkdir(exist_ok=True)
    GRIB.write_bytes(content)
    return GRIB


def field(path, name):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={
        "filter_by_keys": {"shortName": name, "typeOfLevel": "isobaricInhPa"},
        "indexpath": ""})
    da = ds[list(ds.data_vars)[0]]
    if not set(LEVELS).issubset(set(da.isobaricInhPa.values)):
        raise ValueError(f"Incomplete GFS levels for {name}")
    return da.sel(isobaricInhPa=list(LEVELS)).load()


def score_step(path):
    t, q, gh = (field(path, name) for name in ("t", "q", "gh"))
    u, v, w = (field(path, name) for name in ("u", "v", "w"))
    t_c = t - 273.15
    get = lambda da, level: da.sel(isobaricInhPa=level).values
    lapse = (get(t_c, 700) - get(t_c, 500)) / np.maximum(
        (get(gh, 500) - get(gh, 700)) / 1000, .1)
    shear = np.hypot(get(u, 300) - get(u, 850), get(v, 300) - get(v, 850))
    q850 = get(q, 850)
    e850 = q850 * 850 / (0.622 + 0.378 * q850)
    ratio = np.log(np.maximum(e850, .01) / 6.112)
    td850 = 243.5 * ratio / (17.67 - ratio) + 273.15
    li = approximate_li850(get(t, 850), td850, get(t, 500),
                           get(gh, 850), get(gh, 500))
    wet_c = approximate_wet_bulb_c(t_c, q, t.isobaricInhPa.values)
    wet_da = xr.DataArray(wet_c, coords=t_c.coords, dims=t_c.dims)
    wbz = height_at_isotherm(wet_da, gh, 0)
    h10 = height_at_isotherm(t_c, gh, -10)
    h30 = height_at_isotherm(t_c, gh, -30)
    score = hail_potential_v2(
        shear06=shear, wbz_m=wbz, hgl_depth_m=np.maximum(h30 - h10, 0),
        lapse_700_500=lapse, t500_c=get(t_c, 500),
        mixing_ratio=q850 * 1000, li850=li,
        omega700=get(w, 700), omega500=get(w, 500))
    return t.longitude.values, t.latitude.values, score


def latest_cycle():
    now = datetime.now(timezone.utc)
    anchor = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0,
                         microsecond=0)
    for offset in range(5):
        cycle = anchor - timedelta(hours=offset * 6)
        try:
            score_step(download(cycle, 72))
            return cycle
        except (OSError, ValueError, IndexError) as exc:
            print(f"GFS {cycle:%Y-%m-%d %HZ} incomplete: {exc}")
    raise RuntimeError("No complete GFS 72-hour cycle")


def run(cycle=None, hours=HOURS):
    cycle = cycle or latest_cycle()
    peak = None
    peak_hour = None
    for hour in hours:
        lon, lat, score = score_step(download(cycle, hour))
        if peak is None:
            peak = score
        else:
            peak = np.fmax(peak, score)
        if peak_hour is None or np.nanmax(score) > peak_hour[1]:
            peak_hour = (hour, np.nanmax(score))
        print(f"GFS hail +{hour}h loaded", flush=True)
    fig = plt.figure(figsize=(13, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([20, 65, 10, 45], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=.8)
    ax.add_feature(cfeature.BORDERS, linewidth=.6)
    image = ax.contourf(lon, lat, np.ma.masked_less(peak, 20),
                        levels=np.arange(20, 101, 10), cmap="turbo",
                        extend="max", transform=ccrs.PlateCarree())
    fig.colorbar(image, ax=ax, shrink=.8,
                 label="Experimental GFS Large Hail Environment (0–100)")
    valid = cycle + timedelta(hours=peak_hour[0])
    ax.set_title("GFS Experimental Large Hail Environment | Middle East\n"
                 f"Run {cycle:%d %b %Y %HZ} | Window +6 to +72h | "
                 f"Domain peak +{peak_hour[0]}h ({valid:%d %b %HZ})\n"
                 "HGL + WBZ + shear, with lapse, moisture, LI and ascent; "
                 "not observed hail or hail diameter")
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    run()
