"""Download 0-72 h GEFS precipitation for the Middle East.

The downloader uses NOAA's GRIB filter, so only the requested region and
field are transferred.  It is intentionally independent from the operational
ECMWF severe-weather workflow.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import numpy as np
import requests
import xarray as xr


DATA_DIR = Path("data/gefs_precip")
OUTPUT_FILE = Path("data/gefs_precip_ensemble.npz")
STATUS_FILE = Path("output/ensemble_status.json")
STEPS = tuple(range(6, 73, 6))
MEMBERS = ("c00",) + tuple(f"p{i:02d}" for i in range(1, 31))
WEST, EAST, SOUTH, NORTH = 20, 65, 10, 45
FILTER = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl"


def candidate_cycles():
    now = datetime.now(timezone.utc)
    anchor = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)
    for offset in range(0, 37, 6):
        yield anchor - timedelta(hours=offset)


def request_params(cycle, member, step):
    prefix = "gec" if member == "c00" else "gep"
    number = "00" if member == "c00" else member[1:]
    filename = f"{prefix}{number}.t{cycle:%H}z.pgrb2a.0p50.f{step:03d}"
    directory = f"/gefs.{cycle:%Y%m%d}/{cycle:%H}/atmos/pgrb2ap5"
    return {
        "file": filename,
        "lev_surface": "on",
        "var_APCP": "on",
        "subregion": "",
        "leftlon": WEST,
        "rightlon": EAST,
        "toplat": NORTH,
        "bottomlat": SOUTH,
        "dir": directory,
    }


def find_cycle():
    for cycle in candidate_cycles():
        if datetime.now(timezone.utc) - cycle < timedelta(hours=6):
            continue
        try:
            # The final member and forecast hour must exist before a cycle is selected.
            response = requests.get(
                FILTER, params=request_params(cycle, "p30", 72), timeout=45
            )
            if response.ok and len(response.content) > 1000 and b"GRIB" in response.content[:32]:
                return cycle
        except requests.RequestException as exc:
            print(f"Cycle probe failed: {cycle}: {exc}")
    raise RuntimeError("No complete recent GEFS cycle was found")


def download_one(cycle, member, step):
    cycle_dir = DATA_DIR / cycle.strftime("%Y%m%d%H")
    cycle_dir.mkdir(parents=True, exist_ok=True)
    target = cycle_dir / f"{member}_f{step:03d}.grib2"
    if target.exists() and target.stat().st_size > 1000:
        return member, step, target
    for attempt in range(6):
        try:
            response = requests.get(
                FILTER, params=request_params(cycle, member, step), timeout=90
            )
            if response.ok and len(response.content) > 1000 and b"GRIB" in response.content[:32]:
                target.write_bytes(response.content)
                return member, step, target
        except requests.RequestException as exc:
            print(f"Retry {member} f{step:03d}: {exc}")
        time.sleep(min(30, 2 ** (attempt + 1)))
    raise RuntimeError(f"GEFS download failed: {member} f{step:03d}")


def read_apcp(path):
    ds = xr.open_dataset(
        path, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "tp"}, "indexpath": ""},
    )
    da = ds[list(ds.data_vars)[0]].squeeze(drop=True)
    values = np.asarray(da.values, dtype=np.float32)
    # NCEP APCP is normally decoded in kg m-2 (equivalent to mm).
    if float(np.nanmax(values)) < 2.0:
        values *= 1000.0
    return values, np.asarray(da.latitude), np.asarray(da.longitude)


def write_status(cycle, state, detail, members=0):
    STATUS_FILE.parent.mkdir(exist_ok=True)
    STATUS_FILE.write_text(json.dumps({
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle.strftime("%Y-%m-%d %H UTC") if cycle else None,
        "state": state,
        "detail": detail,
        "systems": {"GEFS": {"available_members": members, "target_members": 31}},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    cycle = find_cycle()
    write_status(cycle, "downloading", "GEFS precipitation members", 0)
    files = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_one, cycle, m, s) for m in MEMBERS for s in STEPS]
        for future in as_completed(futures):
            member, step, path = future.result()
            files[member, step] = path

    totals = []
    latitude = longitude = None
    for member in MEMBERS:
        # GEFS APCP messages in this product are interval accumulations.
        # Add every 6-hour period to obtain the complete 0-72 h total.
        intervals = []
        for step in STEPS:
            values, latitude, longitude = read_apcp(files[member, step])
            intervals.append(values)
        totals.append(np.sum(intervals, axis=0, dtype=np.float32))
    stack = np.stack(totals, axis=0)
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE, precipitation_mm=stack, members=np.asarray(MEMBERS),
        latitude=latitude, longitude=longitude,
        init_time=np.datetime64(cycle.replace(tzinfo=None)), forecast_hour=72,
    )
    write_status(cycle, "ready", "GEFS 0-72 h precipitation ensemble", len(MEMBERS))
    print(f"Saved {len(MEMBERS)} GEFS members to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

