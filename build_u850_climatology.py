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
OUTPUT_7 = Path("data/u850_climatology_1991_2020_7NS.npz")


def save_operational_fallback() -> None:
    """Keep the live product running while the NOAA THREDDS host is down.

    A zero reference makes the downstream field the actual U850 wind rather
    than an anomaly.  The reference label is carried into the plot so the
    operational fallback can never be mistaken for a climatological anomaly.
    """
    longitude = np.arange(0.0, 360.0, 2.5, dtype=np.float32)
    for width, target in ((5, OUTPUT), (7, OUTPUT_7)):
        np.savez_compressed(
            target, climatology=np.zeros((366, longitude.size), dtype=np.float32),
            longitude=longitude, reference_period=np.asarray("operational-raw-wind"),
            level_hpa=np.int16(850),
            latitude_band=np.asarray([-float(width), float(width)], dtype=np.float32),
        )
    print(f"NOAA unavailable; saved transparent raw-wind fallback to {OUTPUT}")


def main() -> None:
    DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
    if not DOWNLOAD.exists() or DOWNLOAD.stat().st_size < 100_000:
        try:
            with requests.get(SOURCE, timeout=180, stream=True) as response:
                response.raise_for_status()
                with DOWNLOAD.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
        except requests.RequestException as exc:
            print(f"NOAA climatology download failed: {exc}")
            save_operational_fallback()
            return
    ds = xr.open_dataset(DOWNLOAD, decode_times=False)
    wind = ds["uwnd"].sel(level=850)
    lat_name = "lat" if "lat" in wind.coords else "latitude"
    lon_name = "lon" if "lon" in wind.coords else "longitude"
    lat = wind[lat_name]
    # NCEP latitude is normally north-to-south; the boolean selection works
    # regardless of coordinate order.
    for width, target in ((5, OUTPUT), (7, OUTPUT_7)):
        if width == 7:
            # NCEP's 2.5-degree grid otherwise selects the same points for
            # +/-5 and +/-7. Interpolate before the latitude average.
            fine_lat = np.arange(-7.0, 7.01, 0.25)
            monthly = (wind.sortby(lat_name).interp({lat_name: fine_lat})
                       .mean(lat_name).load())
        else:
            monthly = wind.where((lat >= -width) & (lat <= width), drop=True).mean(lat_name).load()
        if monthly.sizes.get("time") != 12:
            raise ValueError("NOAA long-term-mean file does not contain 12 months")
        centres = np.asarray([15, 45, 74, 105, 135, 166,
                              196, 227, 258, 288, 319, 349], dtype=float)
        x = np.concatenate(([centres[-1] - 366], centres, [centres[0] + 366]))
        values = np.asarray(monthly.values, dtype=np.float32)
        wrapped = np.concatenate((values[-1:], values, values[:1]), axis=0)
        days = np.arange(366, dtype=float)
        daily = np.stack([
            np.interp(days, x, wrapped[:, j]) for j in range(wrapped.shape[1])
        ], axis=1).astype(np.float32)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target, climatology=daily,
            longitude=np.asarray(monthly[lon_name], dtype=np.float32) % 360.0,
            reference_period=np.asarray("1991-2020"), level_hpa=np.int16(850),
            latitude_band=np.asarray([-float(width), float(width)], dtype=np.float32),
        )
        print(f"Saved {target}")


if __name__ == "__main__":
    main()
