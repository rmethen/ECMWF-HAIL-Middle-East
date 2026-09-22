"""Create the 1991-2020 equatorial U850 daily climatology from NOAA PSL.

The NOAA/NCEP reanalysis is read through OPeNDAP, so only the requested
850-hPa equatorial belt is transferred.  Monthly means are converted to a
smooth 366-day climatology using periodic interpolation.
"""

from pathlib import Path

import numpy as np
import xarray as xr


SOURCE = (
    "https://psl.noaa.gov/thredds/dodsC/"
    "Datasets/ncep.reanalysis.derived/pressure/uwnd.mon.mean.nc"
)
OUTPUT = Path("data/u850_climatology_1991_2020.npz")


def main() -> None:
    ds = xr.open_dataset(SOURCE, decode_times=True)
    wind = ds["uwnd"].sel(level=850, time=slice("1991-01-01", "2020-12-31"))
    lat_name = "lat" if "lat" in wind.coords else "latitude"
    lon_name = "lon" if "lon" in wind.coords else "longitude"
    lat = wind[lat_name]
    # NCEP latitude is normally north-to-south; the boolean selection works
    # regardless of coordinate order.
    wind = wind.where((lat >= -5) & (lat <= 5), drop=True).mean(lat_name)
    monthly = wind.groupby("time.month").mean("time").load()

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
