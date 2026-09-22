"""Create the 1991-2020 equatorial U850 daily climatology from NOAA PSL.

NOAA publishes the 12-month 1991-2020 long-term mean directly.  The compact
file is downloaded once and converted to a smooth 366-day climatology.
"""

from pathlib import Path

import numpy as np
import requests
import xarray as xr


SOURCE = (
    "https://psl.noaa.gov/thredds/fileServer/"
    "Datasets/ncep.reanalysis/Monthlies/pressure/"
    "uwnd.mon.ltm.1991-2020.nc"
)
DOWNLOAD = Path("data/noaa_uwnd_mon_ltm_1991_2020.nc")
OUTPUT = Path("data/u850_climatology_1991_2020.npz")


def main() -> None:
    DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
    if not DOWNLOAD.exists() or DOWNLOAD.stat().st_size < 100_000:
        with requests.get(SOURCE, timeout=180, stream=True) as response:
            response.raise_for_status()
            with DOWNLOAD.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    ds = xr.open_dataset(DOWNLOAD, decode_times=False)
    wind = ds["uwnd"].sel(level=850)
    lat_name = "lat" if "lat" in wind.coords else "latitude"
    lon_name = "lon" if "lon" in wind.coords else "longitude"
    lat = wind[lat_name]
    # NCEP latitude is normally north-to-south; the boolean selection works
    # regardless of coordinate order.
    monthly = wind.where((lat >= -5) & (lat <= 5), drop=True).mean(lat_name).load()
    if monthly.sizes.get("time") != 12:
        raise ValueError("NOAA long-term-mean file does not contain 12 months")

    # Place monthly means at month centres, then interpolate cyclically to all
    # days.  Leap day is retained so indexing stays stable in leap years.
    centres = np.asarray([15, 45, 74, 105, 135, 166,
                          196, 227, 258, 288, 319, 349], dtype=float)
    x = np.concatenate(([centres[-1] - 366], centres, [centres[0] + 366]))
    values = np.asarray(monthly.values, dtype=np.float32)
    wrapped = np.concatenate((values[-1:], values, values[:1]), axis=0)
    days = np.arange(366, dtype=float)
    daily = np.stack([
        np.interp(days, x, wrapped[:, j]) for j in range(wrapped.shape[1])
    ], axis=1).astype(np.float32)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        climatology=daily,
        longitude=np.asarray(monthly[lon_name], dtype=np.float32) % 360.0,
        reference_period=np.asarray("1991-2020"),
        level_hpa=np.int16(850),
        latitude_band=np.asarray([-5.0, 5.0], dtype=np.float32),
    )
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
