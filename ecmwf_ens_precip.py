"""Download ECMWF ENS 0-72 h precipitation for the Middle East."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ecmwf.opendata import Client
import numpy as np
import xarray as xr


DATA_DIR = Path("data/ecmwf_ens")
PF_FILE = DATA_DIR / "ens_pf_tp_f072.grib2"
CF_FILE = DATA_DIR / "ens_cf_tp_f072.grib2"
OUTPUT_FILE = Path("data/ecmwf_ens_precip_ensemble.npz")
STATUS_FILE = Path("output/ecmwf_ens_status.json")
NORTH, WEST, SOUTH, EAST = 45, 20, 10, 65
PERTURBED_MEMBERS = list(range(1, 51))


def download_fields():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Use the AWS mirror to avoid congestion limits on the primary portal.
    client = Client(source="aws", model="ifs", resol="0p25")
    request = {
        "stream": "enfo",
        "step": 72,
        "param": "tp",
    }
    print("Downloading ECMWF ENS control member...")
    client.retrieve(type="cf", target=str(CF_FILE), **request)
    print("Downloading 50 ECMWF ENS perturbed members...")
    client.retrieve(
        type="pf", number=PERTURBED_MEMBERS, target=str(PF_FILE), **request
    )


def open_tp(path, data_type):
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"shortName": "tp", "dataType": data_type},
            "indexpath": "",
        },
    )
    field = ds["tp"] if "tp" in ds else ds[list(ds.data_vars)[0]]
    # The open-data mirrors provide global fields; crop after decoding.
    lat = field.latitude
    lat_slice = slice(NORTH, SOUTH) if lat[0] > lat[-1] else slice(SOUTH, NORTH)
    return field.sel(latitude=lat_slice, longitude=slice(WEST, EAST))


def to_mm(values):
    values = np.asarray(values, dtype=np.float32)
    # ECMWF total precipitation is encoded in metres.
    return np.maximum(values * 1000.0, 0.0)


def main():
    download_fields()
    control = open_tp(CF_FILE, "cf").squeeze(drop=True)
    perturbed = open_tp(PF_FILE, "pf").squeeze(drop=True)
    if "number" not in perturbed.dims:
        raise RuntimeError("ECMWF perturbed-member dimension was not decoded")
    stack = np.concatenate(
        [to_mm(control.values)[None, ...], to_mm(perturbed.values)], axis=0
    )
    if stack.shape[0] != 51:
        raise RuntimeError(f"Expected 51 ENS members, decoded {stack.shape[0]}")

    init_value = control.coords.get("time", np.datetime64("NaT")).values
    init_time = np.asarray(init_value).reshape(-1)[0]
    members = np.asarray(["cf"] + [f"pf{i:02d}" for i in PERTURBED_MEMBERS])
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        precipitation_mm=stack,
        members=members,
        latitude=np.asarray(control.latitude),
        longitude=np.asarray(control.longitude),
        init_time=init_time,
        forecast_hour=72,
    )
    STATUS_FILE.parent.mkdir(exist_ok=True)
    STATUS_FILE.write_text(
        json.dumps(
            {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "state": "ready",
                "system": "ECMWF ENS",
                "available_members": int(stack.shape[0]),
                "target_members": 51,
                "forecast_hour": 72,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {stack.shape[0]} ECMWF ENS members to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
