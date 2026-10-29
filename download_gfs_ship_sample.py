"""Download and validate one genuine GFS +6h profile GRIB for SHIP.

No hail map is published by this smoke test.
"""
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from gfs_ship_inputs import pressure_profile_url

OUT = Path("data/gfs_ship_f006.grib2")


def download_one(cycle, output=OUT):
    url = pressure_profile_url(cycle, 6)
    with urlopen(Request(url, headers={"User-Agent": "ECMWF-HAIL-SHIP-research/1.0"}),
                 timeout=180) as response:
        first = response.read(4)
        if first != b"GRIB":
            raise ValueError("NOMADS did not return GRIB2 data")
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".part")
        try:
            with temp.open("wb") as handle:
                handle.write(first)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            if temp.stat().st_size < 10000:
                raise ValueError("GFS GRIB download unexpectedly small")
            temp.replace(output)
        finally:
            temp.unlink(missing_ok=True)
    return output


def latest_available(max_cycles=8):
    now = datetime.now(timezone.utc)
    anchor = now.replace(hour=now.hour // 6 * 6, minute=0, second=0,
                         microsecond=0)
    for offset in range(max_cycles):
        cycle = anchor - timedelta(hours=offset * 6)
        try:
            path = download_one(cycle)
            print(f"GFS cycle {cycle:%Y-%m-%d %HZ}, forecast +6h: "
                  f"{path} ({path.stat().st_size} bytes)")
            return cycle, path
        except (OSError, ValueError) as exc:
            print(f"Skipping {cycle:%Y-%m-%d %HZ}: {exc}", flush=True)
    raise RuntimeError("No usable GFS cycle downloaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", help="YYYYMMDDHH UTC; default latest available")
    args = parser.parse_args()
    if args.cycle:
        cycle = datetime.strptime(args.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        print(download_one(cycle))
    else:
        latest_available()
