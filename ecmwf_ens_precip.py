"""Download ECMWF ENS 0-72 h precipitation for the Middle East."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from ecmwf.opendata import Client
import numpy as np
import xarray as xr


DATA_DIR = Path("data/ecmwf_ens")
PF_FILE = DATA_DIR / "ens_pf_tp_f072.grib2"
OUTPUT_FILE = Path("data/ecmwf_ens_precip_ensemble.npz")
STATUS_FILE = Path("output/ecmwf_ens_status.json")
NORTH, WEST, SOUTH, EAST = 45, 20, 10, 65
ENS_NUMBERS = list(range(1, 51))


def download_fields():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    anchor = now.replace(hour=(now.hour // 12) * 12, minute=0, second=0, microsecond=0)
    for offset in (0, 12):
        cycle = anchor - timedelta(hours=offset)
        # Probe the actual mirrors once a cycle is six hours old; a hard
        # eight-hour gate skips already complete fields during manual updates.
        # The decoded initialization time is still checked before publication.
        if now - cycle < timedelta(hours=6) or now - cycle > timedelta(hours=15):
            continue
        for source in ("azure", "ecmwf"):
            try:
                client = Client(source=source, model="ifs", resol="0p25")
                client.retrieve(date=cycle.strftime("%Y%m%d"), time=cycle.hour,
                                stream="enfo", step=72, param="tp", type="pf",
                                number=ENS_NUMBERS, target=str(PF_FILE) + ".part")
                Path(str(PF_FILE) + ".part").replace(PF_FILE)
                return
            except Exception as exc:
                print(f"ENS {cycle.isoformat()} {source} unavailable: {exc}")
    raise RuntimeError("No complete recent ECMWF ENS cycle on either mirror")


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
    ensemble = open_tp(PF_FILE, "pf").squeeze(drop=True)
    if "number" not in ensemble.dims:
        raise RuntimeError("ECMWF member dimension was not decoded")
    stack = to_mm(ensemble.values)
    if stack.shape[0] != 50:
        raise RuntimeError(f"Expected 50 ENS members, decoded {stack.shape[0]}")

    init_value = ensemble.coords.get("time", np.datetime64("NaT")).values
    init_time = np.asarray(init_value).reshape(-1)[0]
    if np.isnat(init_time):
        raise RuntimeError("ECMWF ENS data has no forecast initialization time")
    init_utc = datetime.fromisoformat(np.datetime_as_string(init_time, unit="s")).replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - init_utc > timedelta(hours=15):
        raise RuntimeError(f"ECMWF ENS cycle {init_utc.isoformat()} is stale; refusing to publish an old map")
    members = np.asarray([f"pf{i:02d}" for i in ENS_NUMBERS])
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        precipitation_mm=stack,
        members=members,
        latitude=np.asarray(ensemble.latitude),
        longitude=np.asarray(ensemble.longitude),
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
                "target_members": 50,
                "forecast_hour": 72,
                "cycle_utc": init_utc.isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {stack.shape[0]} ECMWF ENS members to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
