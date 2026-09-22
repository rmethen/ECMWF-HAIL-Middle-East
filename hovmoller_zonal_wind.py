"""Build an equatorial 850-hPa zonal-wind Hovmoeller data set.

The output contract is deliberately small and model-agnostic.  A downloader
can save analysis and forecast wind fields on a common longitude grid, while
the plotting program consumes the resulting NPZ file.

Expected input arrays in ``data/hovmoller_u850_raw.npz``:

* longitude: 1-D degrees east (0..360)
* time: 1-D numpy datetime64 values
* u850: time x latitude x longitude, metres per second
* latitude: 1-D degrees north, spanning at least 5S..5N
* climatology: day-of-year x longitude, 1991-2020 mean u850 averaged 5S..5N
* forecast_start: scalar datetime64 marking the first forecast time

The script averages 5S-5N and subtracts the matching daily climatology.
"""

from pathlib import Path

import numpy as np


RAW_FILE = Path("data/hovmoller_u850_raw.npz")
OUT_FILE = Path("data/hovmoller_u850_anomaly.npz")


def day_of_year(times: np.ndarray) -> np.ndarray:
    days = times.astype("datetime64[D]")
    years = times.astype("datetime64[Y]")
    return (days - years).astype(int)


def periodic_interp(source_lon: np.ndarray, source: np.ndarray,
                    target_lon: np.ndarray) -> np.ndarray:
    """Interpolate longitude fields across the Greenwich seam."""
    source_lon = np.asarray(source_lon, dtype=float) % 360.0
    order = np.argsort(source_lon)
    x = source_lon[order]
    fields = source[:, order]
    x_wrap = np.concatenate(([x[-1] - 360.0], x, [x[0] + 360.0]))
    return np.stack([
        np.interp(target_lon % 360.0, x_wrap,
                  np.concatenate(([row[-1]], row, [row[0]])))
        for row in fields
    ]).astype(np.float32)


def main() -> None:
    raw = np.load(RAW_FILE)
    lon = np.asarray(raw["longitude"], dtype=np.float32) % 360.0
    lat = np.asarray(raw["latitude"], dtype=np.float32)
    times = raw["time"].astype("datetime64[h]")
    u850 = np.asarray(raw["u850"], dtype=np.float32)
    clim = np.asarray(raw["climatology"], dtype=np.float32)
    clim_lon = np.asarray(raw["climatology_longitude"], dtype=np.float32)

    belt = (lat >= -5.0) & (lat <= 5.0)
    if not np.any(belt):
        raise ValueError("Input latitude grid does not cover 5S-5N")
    equatorial_u = np.nanmean(u850[:, belt, :], axis=1)
    doy = np.clip(day_of_year(times), 0, clim.shape[0] - 1)
    daily_clim = periodic_interp(clim_lon, clim, lon)
    anomaly = equatorial_u - daily_clim[doy]

    order = np.argsort(lon)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_FILE,
        longitude=lon[order],
        time=times,
        u_anomaly=anomaly[:, order],
        forecast_start=raw["forecast_start"].astype("datetime64[h]"),
        level_hpa=np.int16(850),
        latitude_band=np.asarray([-5.0, 5.0], dtype=np.float32),
    )
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
