"""
ECMWF Open Data loader for Experimental Hail Potential Index V2
Middle East | Forecast 0–72 h
"""

from pathlib import Path
from ecmwf.opendata import Client

# Middle East domain
NORTH = 45
WEST = 20
SOUTH = 10
EAST = 65

# Forecast hours
STEPS = list(range(0, 73, 3))

# Pressure levels needed for hail diagnostics
LEVELS = [
    1000,
    925,
    850,
    700,
    600,
    500,
    400,
    300,
]

# Variables available in ECMWF Open Data
PARAMS = [
    "t",   # temperature
    "u",   # zonal wind
    "v",   # meridional wind
    "q",   # specific humidity
    "w",   # vertical velocity
    "gh",  # geopotential height
]


def download_ecmwf_hail_fields():
    """
    Download real ECMWF IFS pressure-level forecast fields
    required for the experimental hail diagnostic.
    """

    Path("data").mkdir(exist_ok=True)

    target = Path("data/ecmwf_hail_0_72h.grib2")

    client = Client(
        source="ecmwf",
        model="ifs",
        resol="0p25",
    )

    print("Downloading ECMWF IFS hail diagnostic fields...")
    print("Forecast range: 0–72 h")
    print("Domain: Middle East")

    client.retrieve(
        type="fc",
        stream="oper",
        step=STEPS,
        param=PARAMS,
        levelist=LEVELS,
        target=str(target),
    )

    print("ECMWF download complete:")
    print(target)

    return target


if __name__ == "__main__":
    download_ecmwf_hail_fields()
