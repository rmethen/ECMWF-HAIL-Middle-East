"""Download ECMWF Open Data fields for the severe-weather diagnostics."""

from pathlib import Path
from ecmwf.opendata import Client

NORTH, WEST, SOUTH, EAST = 45, 20, 10, 65
STEPS = list(range(0, 73, 3))
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300]
PRESSURE_PARAMS = ["t", "u", "v", "q", "w", "gh"]
SURFACE_PARAMS = ["2t", "2d", "10u", "10v", "fg10", "msl", "tp"]


def download_ecmwf_hail_fields():
    Path("data").mkdir(exist_ok=True)
    pressure_target = Path("data/ecmwf_hail_0_72h.grib2")
    surface_target = Path("data/ecmwf_surface_0_72h.grib2")
    client = Client(source="ecmwf", model="ifs", resol="0p25")

    print("Downloading ECMWF IFS pressure-level fields...")
    client.retrieve(
        type="fc", stream="oper", step=STEPS, param=PRESSURE_PARAMS,
        levelist=LEVELS, area=[NORTH, WEST, SOUTH, EAST],
        target=str(pressure_target),
    )
    print("Downloading ECMWF IFS surface fields...")
    client.retrieve(
        type="fc", stream="oper", step=STEPS, param=SURFACE_PARAMS,
        area=[NORTH, WEST, SOUTH, EAST], target=str(surface_target),
    )
    print("ECMWF downloads complete")
    return pressure_target


if __name__ == "__main__":
    download_ecmwf_hail_fields()
