"""Fetch the GFS surface boundary fields required for a physical SHIP profile."""
from datetime import datetime
from urllib.parse import urlencode
from gfs_ship_inputs import BASE, DOMAIN

def surface_url(cycle: datetime, forecast_hour: int) -> str:
    if cycle.tzinfo is None or cycle.utcoffset().total_seconds() != 0:
        raise ValueError("UTC cycle required")
    params = {
        "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{forecast_hour:03d}",
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
        "subregion": "", **DOMAIN,
        "var_PRES": "on", "lev_surface": "on",
        "var_HGT": "on",
    }
    return BASE + "?" + urlencode(params)
