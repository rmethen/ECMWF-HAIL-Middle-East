"""Download ECMWF Open Data fields for the severe-weather diagnostics."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import time
from ecmwf.opendata import Client

NORTH, WEST, SOUTH, EAST = 45, 20, 10, 65
STEPS = list(range(0, 73, 6)) + list(range(84, 169, 12))
# The ECMWF 06Z and 18Z IFS control forecasts have a shorter horizon.
SHORT_STEPS = list(range(0, 73, 6))
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
PRESSURE_PARAMS = ["t", "u", "v", "q", "w", "gh"]
SNOWFALL_FILE = Path("data/ecmwf_snowfall.grib2")
SURFACE_PARAMS = ["2t", "2d", "10u", "10v", "msl", "tp", "mucape"]


def download_ecmwf_hail_fields():
    Path("data").mkdir(exist_ok=True)
    pressure_target = Path("data/ecmwf_hail_0_72h.grib2")
    surface_target = Path("data/ecmwf_surface_0_72h.grib2")
    SNOWFALL_FILE.unlink(missing_ok=True)
    snowfall_part = SNOWFALL_FILE.with_suffix(".part")
    now = datetime.now(timezone.utc)
    # IFS 00/06/12/18 are valid cycles. Prefer the freshest complete
    # cycle; avoid requesting long lead times absent from 06/18 products.
    anchor = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)
    for offset in (0, 6, 12, 18, 24):
        cycle = anchor - timedelta(hours=offset)
        if now - cycle < timedelta(hours=6) or now - cycle > timedelta(hours=24):
            continue
        steps = SHORT_STEPS if cycle.hour in (6, 18) else STEPS
        for source in ("azure", "ecmwf"):
            try:
                client = Client(source=source, model="ifs", resol="0p25")
                common = dict(date=cycle.strftime("%Y%m%d"), time=cycle.hour,
                              type="fc", stream="oper", step=steps,
                              area=[NORTH, WEST, SOUTH, EAST])
                client.retrieve(param=PRESSURE_PARAMS, levelist=LEVELS,
                                target=str(pressure_target) + ".part", **common)
                client.retrieve(param=SURFACE_PARAMS,
                                target=str(surface_target) + ".part", **common)
                Path(str(pressure_target) + ".part").replace(pressure_target)
                Path(str(surface_target) + ".part").replace(surface_target)
                try:
                    client.retrieve(param="sf", target=str(snowfall_part), **common)
                    snowfall_part.replace(SNOWFALL_FILE)
                except Exception as exc:
                    snowfall_part.unlink(missing_ok=True)
                    print(f"ECMWF snowfall unavailable for {cycle.isoformat()}: {exc}")
                status = dict(cycle_utc=cycle.isoformat(), forecast_steps=steps,
                              updated_utc=now.isoformat(), state="ready")
                Path("data/hail_download_status.json").write_text(json.dumps(status, indent=2))
                print(f"Downloaded matching ECMWF fields: {cycle.isoformat()} from {source}")
                return pressure_target
            except Exception as exc:
                print(f"ECMWF {cycle.isoformat()} {source} unavailable: {exc}")
                time.sleep(2)
    raise RuntimeError("No complete ECMWF 00/06/12/18 UTC cycle within 24 hours; retaining published maps")


if __name__ == "__main__":
    download_ecmwf_hail_fields()

