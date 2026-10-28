"""Conservative ECMWF–GFS agreement in the large-hail environment index.

The index at each valid time is the lower of two independently computed V2
scores. It is an uncalibrated agreement diagnostic, not a hail probability.
"""

from datetime import datetime, timezone
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gfs_hail_map import download, score_step
from hail_index import hail_potential_v2


DATA = Path("data/hail_diagnostics.npz")
OUTPUT = Path("output/ECMWF_GFS_HAIL_AGREEMENT_LATEST.png")


def ecmwf_scores(data):
    return hail_potential_v2(
        shear06=data["shear_850_300"], wbz_m=data["wbz_m"],
        hgl_depth_m=data["hgl_depth_m"],
        lapse_700_500=data["lapse_700_500"], t500_c=data["t500_c"],
        mixing_ratio=data["moisture_850"], li850=data["li850"],
        omega700=data["omega_700"], omega500=data["omega_500"],
    )


def step_hours(steps):
    if np.issubdtype(steps.dtype, np.timedelta64):
        return (steps / np.timedelta64(1, "h")).astype(int)
    return steps.astype(int)


def aligned_lower_score(ecmwf, elat, elon, gfs, glat, glon):
    """Compare scores after spatial interpolation to the ECMWF grid."""
    regridded = xr.DataArray(
        gfs, dims=("latitude", "longitude"),
        coords={"latitude": glat, "longitude": glon},
    ).interp(latitude=elat, longitude=elon, method="linear").values
    return np.minimum(ecmwf, regridded)


def main():
    with np.load(DATA) as data:
        init = np.datetime64(data["init_time"])
        if np.isnat(init):
            raise ValueError("ECMWF run time is missing")
        # Both models must start at the same 00/12 UTC cycle to match +hours.
        cycle = datetime.fromisoformat(np.datetime_as_string(init, unit="s"))
        cycle = cycle.replace(tzinfo=timezone.utc)
        elat, elon = data["latitude"], data["longitude"]
        scores = ecmwf_scores(data)
        hours = step_hours(data["steps"])
        agreed = np.full(scores.shape[1:], np.nan)
        matched = []
        for hour in range(6, 73, 6):
            locations = np.where(hours == hour)[0]
            if locations.size != 1:
                raise ValueError(f"ECMWF +{hour}h is unavailable")
            glon, glat, gscore = score_step(download(cycle, hour))
            current = aligned_lower_score(
                scores[locations[0]], elat, elon, gscore, glat, glon)
            agreed = np.fmax(agreed, current)
            matched.append(hour)
            print(f"Matched ECMWF and GFS +{hour}h", flush=True)

    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([20, 65, 10, 45], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=.8)
    ax.add_feature(cfeature.BORDERS, linewidth=.6)
    image = ax.contourf(elon, elat, np.ma.masked_less(agreed, 20),
                        levels=np.arange(20, 101, 10), cmap="turbo",
                        extend="max", transform=ccrs.PlateCarree())
    fig.colorbar(image, ax=ax, shrink=.8,
                 label="Experimental ECMWF–GFS hail agreement (0–100)")
    ax.set_title("ECMWF + GFS Large Hail Environment Agreement | Middle East\n"
                 f"Both runs {cycle:%d %b %Y %HZ} | Matching valid times "
                 f"+{matched[0]} to +{matched[-1]}h\n"
                 "Lower score at each matched hour, then local maximum; "
                 "not observed hail or calibrated probability")
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
