"""Download ECMWF Open Data fields for the severe-weather diagnostics."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import time
from ecmwf.opendata import Client

NORTH, WEST, SOUTH, EAST = 45, 20, 10, 65
STEPS = list(range(0, 73, 6)) + list(range(84, 169, 12))
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
PRESSURE_PARAMS = ["t", "u", "v", "q", "w", "gh"]
SURFACE_PARAMS = ["2t", "2d", "10u", "10v", "msl", "tp", "mucape"]


def download_ecmwf_hail_fields():
    Path("data").mkdir(exist_ok=True)
    pressure_target = Path("data/ecmwf_hail_0_72h.grib2")
    surface_target = Path("data/ecmwf_surface_0_72h.grib2")
    now = datetime.now(timezone.utc)
    anchor = now.replace(hour=(now.hour // 12) * 12, minute=0, second=0, microsecond=0)
    for offset in (0, 12, 24):
        cycle = anchor - timedelta(hours=offset)
        if now - cycle < timedelta(hours=8) or now - cycle > timedelta(hours=24):
            continue
        for source in ("azure", "ecmwf"):
            try:
                client = Client(source=source, model="ifs", resol="0p25")
                common = dict(date=cycle.strftime("%Y%m%d"), time=cycle.hour,
                              type="fc", stream="oper", step=STEPS,
                              area=[NORTH, WEST, SOUTH, EAST])
                client.retrieve(param=PRESSURE_PARAMS, levelist=LEVELS,
                                target=str(pressure_target) + ".part", **common)
                client.retrieve(param=SURFACE_PARAMS,
                                target=str(surface_target) + ".part", **common)
                Path(str(pressure_target) + ".part").replace(pressure_target)
                Path(str(surface_target) + ".part").replace(surface_target)
                status = dict(cycle_utc=cycle.isoformat(),
                              updated_utc=now.isoformat(), state="ready")
                Path("data/hail_download_status.json").write_text(json.dumps(status, indent=2))
                print(f"Downloaded matching ECMWF fields: {cycle.isoformat()} from {source}")
                return pressure_target
            except Exception as exc:
                print(f"ECMWF {cycle.isoformat()} {source} unavailable: {exc}")
                time.sleep(2)
    raise RuntimeError("No complete ECMWF 00/12 UTC cycle within 24 hours; retaining published maps")


if __name__ == "__main__":
    download_ecmwf_hail_fields()

