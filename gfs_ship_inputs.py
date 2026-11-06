"""Independent GFS input requests for experimental SHIP.

The existing GFS V2 GRIB does not include the MU parcel or near-surface
profile needed for SHIP. Keep downloads separate and fail closed.
"""
from datetime import datetime, timezone
from urllib.parse import urlencode

BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
DOMAIN = dict(leftlon=20, rightlon=65, bottomlat=10, toplat=45)
PRESSURE_LEVELS = (1000, 975, 950, 925, 900, 850, 800, 750, 700, 650,
                   600, 550, 500, 450, 400, 350, 300, 250, 200)


def pressure_profile_url(cycle: datetime, forecast_hour: int) -> str:
    """Request same-cycle T, q, geopotential height, u/v profile.

    This is an input only: MU parcel search, CAPE integration and AGL shear
    interpolation must be validated before any SHIP score is published.
    """
    if cycle.tzinfo is None or cycle.utcoffset().total_seconds() != 0:
        raise ValueError("GFS cycle must be timezone-aware UTC")
    if cycle.minute or cycle.second or cycle.hour not in (0, 6, 12, 18):
        raise ValueError("GFS cycle must be a synoptic 00/06/12/18 UTC cycle")
    if not 0 <= forecast_hour <= 72 or forecast_hour % 6:
        raise ValueError("Forecast hour must be 0..72 in 6-hour increments")
    params = {
        "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{forecast_hour:03d}",
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
        "subregion": "",
        **DOMAIN,
    }
    for name in ("TMP", "SPFH", "HGT", "UGRD", "VGRD"):
        params[f"var_{name}"] = "on"
    for level in PRESSURE_LEVELS:
        params[f"lev_{level}_mb"] = "on"
    return BASE + "?" + urlencode(params)


def input_status() -> dict:
    """Document outstanding scientific prerequisites for a real SHIP map."""
    return {
        "pressure_profile_request": "implemented, not live-tested",
        "surface_pressure_and_terrain": "not implemented",
        "most_unstable_parcel_search": "not implemented",
        "mucape_integration": "not implemented",
        "0_6km_agl_bulk_shear": "not implemented",
        "ship_formula_and_caps": "pending source verification",
        "image_publication": "blocked",
    }
