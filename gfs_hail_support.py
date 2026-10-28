"""Optional GFS simulated storm signal aligned to the ECMWF hail peak time."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import xarray as xr


BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"


def _read(path, short_name, level):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={
        "filter_by_keys": {"shortName": short_name, "typeOfLevel": level,
                           "stepType": "instant"}, "indexpath": ""})
    return ds[list(ds.data_vars)[0]].load()


def storm_support(valid_time):
    """Return (longitude, latitude, mask, source label), or raise on missing GFS.

    This is forecast storm support, not observed radar and not hail occurrence.
    """
    target = datetime.fromisoformat(str(valid_time).replace("Z", "+00:00"))
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    anchor = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0,
                         microsecond=0)
    for offset in range(0, 5):
        cycle = anchor - timedelta(hours=offset * 6)
        hour = round((target - cycle).total_seconds() / 10800) * 3
        if not 3 <= hour <= 120 or abs((cycle + timedelta(hours=hour) - target).total_seconds()) > 5400:
            continue
        params = {
            "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}",
            "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
            "var_REFC": "on", "var_PRATE": "on",
            "lev_entire_atmosphere": "on", "lev_surface": "on",
            "subregion": "", "leftlon": 20, "rightlon": 65,
            "bottomlat": 10, "toplat": 45,
        }
        try:
            with urlopen(Request(BASE + "?" + urlencode(params),
                                 headers={"User-Agent": "ECMWF-HAIL-Middle-East/1.0"}),
                         timeout=40) as response:
                content = response.read()
            if not content.startswith(b"GRIB"):
                continue
            path = Path("data/gfs_hail_support.grib2")
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(content)
            refc = _read(path, "refc", "atmosphere")
            rate = _read(path, "prate", "surface") * 3600.0
            signal = ((refc >= 35) & (rate >= 1)).astype(float)
            return (signal.longitude.values, signal.latitude.values,
                    signal.values, f"GFS {cycle:%d %b %HZ} +{hour}h")
        except (OSError, ValueError, IndexError) as exc:
            print(f"GFS hail support {cycle:%Y-%m-%d %HZ} unavailable: {exc}")
    raise RuntimeError("No GFS run matches the ECMWF hail peak valid time")
