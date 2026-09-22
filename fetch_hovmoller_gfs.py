"""Fetch the equatorial 850-hPa GFS wind band used by the Hovmoeller plot.

Run this program on every workflow cycle.  It downloads only UGRD/850 hPa
between 5S and 5N, keeps a rolling analysis archive, and adds the current GFS
forecast through 384 hours.  A separately cached 1991-2020 climatology is
required by ``hovmoller_zonal_wind.py``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import requests
import xarray as xr


FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
CACHE = Path("data/hovmoller_gfs")
ARCHIVE = Path("data/hovmoller_u850_analysis.npz")
CLIM = Path("data/u850_climatology_1991_2020.npz")
OUTPUT = Path("data/hovmoller_u850_raw.npz")
FORECAST_STEPS = tuple(range(0, 385, 6))


def candidate_cycles():
    now = datetime.now(timezone.utc)
    anchor = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0,
                         microsecond=0)
    for lag in range(0, 37, 6):
        yield anchor - timedelta(hours=lag)


def params(cycle: datetime, step: int) -> dict[str, object]:
    return {
        "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{step:03d}",
        "lev_850_mb": "on",
        "var_UGRD": "on",
        "subregion": "",
        "leftlon": 0,
        "rightlon": 360,
        "toplat": 5,
        "bottomlat": -5,
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
    }


def valid_grib(response: requests.Response) -> bool:
    return response.ok and len(response.content) > 1000 and response.content[:4] == b"GRIB"


def find_cycle() -> datetime:
    for cycle in candidate_cycles():
        response = requests.get(FILTER, params=params(cycle, 0), timeout=45)
        if valid_grib(response):
            return cycle
    raise RuntimeError("No complete GFS cycle found")


def fetch(cycle: datetime, step: int) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / f"gfs_{cycle:%Y%m%d%H}_f{step:03d}.grib2"
    if target.exists() and target.stat().st_size > 1000:
        return target
    for attempt in range(4):
        response = requests.get(FILTER, params=params(cycle, step), timeout=90)
        if valid_grib(response):
            target.write_bytes(response.content)
            return target
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GFS download failed: {cycle:%Y%m%d%H} f{step:03d}")


def read_u850(path: Path):
    ds = xr.open_dataset(
        path, engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"shortName": "u", "typeOfLevel": "isobaricInhPa",
                               "level": 850},
            "indexpath": "",
        },
    )
    da = ds[list(ds.data_vars)[0]].squeeze(drop=True)
    return (np.asarray(da.values, dtype=np.float32),
            np.asarray(da.latitude, dtype=np.float32),
            np.asarray(da.longitude, dtype=np.float32))


def load_analysis():
    if not ARCHIVE.exists():
        return [], []
    old = np.load(ARCHIVE)
    return list(old["time"].astype("datetime64[h]")), list(old["u850"])


def main() -> None:
    cycle = find_cycle()
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(fetch, cycle, step): step for step in FORECAST_STEPS}
        paths = {}
        for job in as_completed(jobs):
            paths[jobs[job]] = job.result()

    forecast_times, forecast_fields = [], []
    latitude = longitude = None
    for step in FORECAST_STEPS:
        field, latitude, longitude = read_u850(paths[step])
        forecast_fields.append(field)
        forecast_times.append(np.datetime64(cycle.replace(tzinfo=None)) + np.timedelta64(step, "h"))

    analysis_times, analysis_fields = load_analysis()
    # The f000 field is the current analysis.  Saving it on every run gradually
    # builds the historical half of the diagram without a large archive pull.
    analysis_times.append(forecast_times[0])
    analysis_fields.append(forecast_fields[0])
    unique = {}
    for t, field in zip(analysis_times, analysis_fields):
        unique[np.datetime64(t, "h")] = field
    cutoff = np.datetime64(cycle.replace(tzinfo=None)) - np.timedelta64(45, "D")
    kept_times = sorted(t for t in unique if t >= cutoff)
    kept_fields = [unique[t] for t in kept_times]
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARCHIVE, time=np.asarray(kept_times),
                        u850=np.stack(kept_fields))

    if not CLIM.exists():
        raise FileNotFoundError(
            f"Missing {CLIM}; install the 1991-2020 daily U850 climatology first"
        )
    climate = np.load(CLIM)
    climatology = climate["climatology"]
    climatology_longitude = climate["longitude"]
    all_times = np.asarray(kept_times + forecast_times[1:])
    all_fields = np.stack(kept_fields + forecast_fields[1:])
    np.savez_compressed(
        OUTPUT, longitude=longitude, latitude=latitude, time=all_times,
        u850=all_fields, climatology=climatology,
        climatology_longitude=climatology_longitude,
        forecast_start=np.asarray(forecast_times[0]),
    )
    print(f"Saved {OUTPUT} from GFS {cycle:%Y-%m-%d %H} UTC")


if __name__ == "__main__":
    main()
