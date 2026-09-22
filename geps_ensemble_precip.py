"""Download Canadian GEPS 0-72 h precipitation for the Middle East."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import numpy as np
import requests
import xarray as xr
import cfgrib


DATA_DIR = Path("data/geps_precip")
GRIB_FILE = DATA_DIR / "geps_apcp_f072_all_members.grib2"
OUTPUT_FILE = Path("data/geps_precip_ensemble.npz")
STATUS_FILE = Path("output/geps_ensemble_status.json")
BASE = "https://dd.weather.gc.ca"
WEST, EAST, SOUTH, NORTH = 20.0, 65.0, 10.0, 45.0


def candidate_cycles():
    now = datetime.now(timezone.utc)
    for day_lag in range(3):
        date = (now - timedelta(days=day_lag)).date()
        for hour in (12, 0):
            cycle = datetime(date.year, date.month, date.day, hour,
                             tzinfo=timezone.utc)
            if cycle <= now:
                yield cycle


def source_url(cycle):
    filename = (
        "CMC_geps-raw_APCP_SFC_0_latlon0p5x0p5_"
        f"{cycle:%Y%m%d%H}_P072_allmbrs.grib2"
    )
    return (
        f"{BASE}/{cycle:%Y%m%d}/WXO-DD/ensemble/geps/grib2/raw/"
        f"{cycle:%H}/072/{filename}"
    )


def find_cycle():
    for cycle in candidate_cycles():
        try:
            response = requests.head(source_url(cycle), timeout=30,
                                     allow_redirects=True)
            if response.ok and int(response.headers.get("content-length", 0)) > 1000:
                return cycle
        except requests.RequestException:
            pass
    raise RuntimeError("No complete recent Canadian GEPS cycle was found")


def download(cycle):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with requests.get(source_url(cycle), timeout=180, stream=True) as response:
                response.raise_for_status()
                with GRIB_FILE.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if GRIB_FILE.stat().st_size > 1000:
                return
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError("Canadian GEPS precipitation download failed")


def open_precipitation():
    # GEPS APCP files have used more than one ecCodes short name over time.
    # Let cfgrib split the GRIB into compatible hypercubes, then select the
    # field that actually contains the ensemble-member and horizontal axes.
    datasets = cfgrib.open_datasets(GRIB_FILE, backend_kwargs={"indexpath": ""})
    candidates = []
    for ds in datasets:
        for name, variable in ds.data_vars.items():
            member = next((d for d in ("number", "realization", "perturbationNumber")
                           if d in variable.dims), None)
            has_lat = "latitude" in variable.coords or "lat" in variable.coords
            has_lon = "longitude" in variable.coords or "lon" in variable.coords
            if member and has_lat and has_lon:
                candidates.append((name, variable))
    if not candidates:
        inventory = [(list(ds.data_vars), dict(ds.sizes)) for ds in datasets]
        raise RuntimeError(f"No ensemble precipitation field found; GRIB inventory: {inventory}")
    name, field = max(candidates, key=lambda item: item[1].size)
    print(f"Decoded GEPS precipitation field {name!r} with dimensions {field.dims}")
    member_dim = next((d for d in ("number", "realization", "perturbationNumber")
                       if d in field.dims), None)
    if member_dim is None:
        raise RuntimeError(f"GEPS member dimension missing; decoded dimensions: {field.dims}")

    lat_name = "latitude" if "latitude" in field.coords else "lat"
    lon_name = "longitude" if "longitude" in field.coords else "lon"
    lat = field[lat_name]
    lat_slice = slice(NORTH, SOUTH) if lat[0] > lat[-1] else slice(SOUTH, NORTH)
    cropped = field.sel({lat_name: lat_slice, lon_name: slice(WEST, EAST)}).squeeze(drop=True)
    values = np.maximum(np.asarray(cropped.values, dtype=np.float32), 0.0)
    units = str(field.attrs.get("units", "")).lower()
    if units in {"m", "metre", "meter"}:
        values *= 1000.0
    members = np.asarray([f"geps{int(v):02d}" for v in cropped[member_dim].values])
    return values, members, np.asarray(cropped[lat_name]), np.asarray(cropped[lon_name])


def main():
    cycle = find_cycle()
    download(cycle)
    rain, members, latitude, longitude = open_precipitation()
    if rain.shape[0] < 20:
        raise RuntimeError(f"Expected at least 20 GEPS members, decoded {rain.shape[0]}")
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE, precipitation_mm=rain, members=members,
        latitude=latitude, longitude=longitude,
        init_time=np.datetime64(cycle.replace(tzinfo=None)), forecast_hour=72,
    )
    STATUS_FILE.parent.mkdir(exist_ok=True)
    STATUS_FILE.write_text(json.dumps({
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "state": "ready", "system": "Canadian GEPS",
        "available_members": int(rain.shape[0]), "forecast_hour": 72,
        "cycle": cycle.strftime("%Y-%m-%d %H UTC"),
    }, indent=2), encoding="utf-8")
    print(f"Saved {rain.shape[0]} Canadian GEPS members to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
