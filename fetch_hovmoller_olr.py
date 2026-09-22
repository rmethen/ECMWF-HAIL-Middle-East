"""Fetch tropical GFS top-of-atmosphere OLR for an operational Hovmoller."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import numpy as np
import requests
import xarray as xr


FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
CACHE = Path("data/hovmoller_olr_gfs")
ARCHIVE = Path("data/hovmoller_olr_analysis.npz")
OUTPUT = Path("data/hovmoller_olr_raw.npz")
FORECAST_STEPS = tuple(range(6, 385, 6))
BACKFILL_HOURS = tuple(range(6, 24 * 8, 6))


def candidate_cycles():
    now = datetime.now(timezone.utc)
    anchor = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0,
                         microsecond=0)
    for lag in range(0, 37, 6):
        yield anchor - timedelta(hours=lag)


def params(cycle: datetime, step: int) -> dict[str, object]:
    return {
        "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{step:03d}",
        "lev_top_of_atmosphere": "on",
        "var_ULWRF": "on",
        "subregion": "",
        "leftlon": 0,
        "rightlon": 360,
        "toplat": 15,
        "bottomlat": -15,
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
    }


def valid_grib(response: requests.Response) -> bool:
    return response.ok and len(response.content) > 1000 and response.content[:4] == b"GRIB"


def find_cycle() -> datetime:
    for cycle in candidate_cycles():
        response = requests.get(FILTER, params=params(cycle, 6), timeout=45)
        if valid_grib(response):
            return cycle
    raise RuntimeError("No complete GFS cycle with top-of-atmosphere ULWRF found")


def fetch(cycle: datetime, step: int = 0) -> Path:
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
    raise RuntimeError(f"GFS OLR download failed: {cycle:%Y%m%d%H} f{step:03d}")


def read_olr(path: Path):
    ds = xr.open_dataset(path, engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    if not ds.data_vars:
        raise RuntimeError(f"No OLR field decoded from {path}")
    da = ds[list(ds.data_vars)[0]].squeeze(drop=True)
    lat = np.asarray(da.latitude, dtype=np.float32)
    weights = np.cos(np.deg2rad(lat))
    field = np.asarray(da.values, dtype=np.float32)
    tropical_mean = np.average(field, axis=0, weights=weights)
    return tropical_mean.astype(np.float32), np.asarray(da.longitude,
                                                        dtype=np.float32)


def load_archive():
    if not ARCHIVE.exists():
        return {}
    old = np.load(ARCHIVE)
    return {np.datetime64(t, "h"): field for t, field
            in zip(old["time"].astype("datetime64[h]"), old["olr"])}


def main() -> None:
    cycle = find_cycle()
    archive = load_archive()
    cutoff = np.datetime64(cycle.replace(tzinfo=None)) - np.timedelta64(45, "D")

    # Seed a useful observed side on the first run; later runs add one analysis
    # every six hours and retain a 45-day rolling archive.
    history_cycles = [cycle - timedelta(hours=h) for h in BACKFILL_HOURS
                      if np.datetime64(cycle.replace(tzinfo=None) - timedelta(hours=h) + timedelta(hours=6), "h")
                      not in archive]
    requests_needed = [(c, 6, "analysis") for c in history_cycles]
    requests_needed += [(cycle, s, "forecast") for s in FORECAST_STEPS]
    paths = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(fetch, c, s): (c, s, kind)
                for c, s, kind in requests_needed}
        for job in as_completed(jobs):
            key = jobs[job]
            try:
                paths[key] = job.result()
            except Exception:
                if key[2] == "forecast":
                    raise

    longitude = None
    for (c, step, kind), path in paths.items():
        if kind != "analysis":
            continue
        field, longitude = read_olr(path)
        archive[np.datetime64(c.replace(tzinfo=None), "h") + np.timedelta64(6, "h")] = field

    forecast_times, forecast_fields = [], []
    for step in FORECAST_STEPS:
        field, longitude = read_olr(paths[(cycle, step, "forecast")])
        valid = np.datetime64(cycle.replace(tzinfo=None), "h") + np.timedelta64(step, "h")
        forecast_times.append(valid)
        forecast_fields.append(field)
    archive[forecast_times[0]] = forecast_fields[0]

    kept_times = sorted(t for t in archive if t >= cutoff)
    kept_fields = [archive[t] for t in kept_times]
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARCHIVE, time=np.asarray(kept_times),
                        olr=np.stack(kept_fields), longitude=longitude)

    all_times = np.asarray(kept_times + forecast_times[1:])
    all_fields = np.stack(kept_fields + forecast_fields[1:])
    np.savez_compressed(
        OUTPUT, time=all_times, olr=all_fields, longitude=longitude,
        forecast_start=np.asarray(forecast_times[0]),
        latitude_band=np.asarray([-15.0, 15.0]),
        source_cycle=np.asarray(cycle.strftime("%Y-%m-%d %H UTC")),
    )
    print(f"Saved {OUTPUT} from GFS {cycle:%Y-%m-%d %H} UTC")


if __name__ == "__main__":
    main()
