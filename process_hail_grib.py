"""
Process ECMWF GRIB2 fields for Experimental Hail Potential Index V2.
"""

from pathlib import Path
import numpy as np
import xarray as xr


GRIB_FILE = Path("data/ecmwf_hail_0_72h.grib2")
OUTPUT_FILE = Path("data/hail_diagnostics.npz")


def open_field(short_name):
    """Open one ECMWF pressure-level variable from the GRIB2 file."""

    return xr.open_dataset(
        GRIB_FILE,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "typeOfLevel": "isobaricInhPa",
                "shortName": short_name,
            },
            "indexpath": "",
        },
    )


def get_level(da, level):
    """Select a pressure level regardless of cfgrib coordinate naming."""

    for coord in ("isobaricInhPa", "level"):
        if coord in da.coords:
            return da.sel({coord: level})

    raise KeyError("Pressure-level coordinate not found")


def main():

    if not GRIB_FILE.exists():
        raise FileNotFoundError(
            f"Missing ECMWF GRIB file: {GRIB_FILE}"
        )

    print("Opening real ECMWF GRIB2 fields...")

    ds_t = open_field("t")
    ds_u = open_field("u")
    ds_v = open_field("v")
    ds_q = open_field("q")
    ds_gh = open_field("gh")
    ds_w = open_field("w")
    t = ds_t["t"]
    u = ds_u["u"]
    v = ds_v["v"]
    q = ds_q["q"]
    gh = ds_gh["gh"]
    w = ds_w["w"]
    # -------------------------------------------------
    # Temperature diagnostics
    # -------------------------------------------------

    t500_c = get_level(t, 500) - 273.15
    t700_c = get_level(t, 700) - 273.15

    z500 = get_level(gh, 500)
    z700 = get_level(gh, 700)

    depth_km = np.maximum(
        (z500 - z700) / 1000.0,
        0.1,
    )

    lapse_700_500 = (
        (t700_c - t500_c) / depth_km
    )

    # -------------------------------------------------
    # Deep-layer wind difference
    # First real proxy for storm-organising shear.
    # -------------------------------------------------

    u850 = get_level(u, 850)
    v850 = get_level(v, 850)

    u300 = get_level(u, 300)
    v300 = get_level(v, 300)

    shear_850_300 = np.sqrt(
        (u300 - u850) ** 2
        +
        (v300 - v850) ** 2
    )
    omega_700 = get_level(w, 700)

    omega_500 = get_level(w, 500)
    # -------------------------------------------------
    # Low-level moisture
    # q kg/kg -> approximate g/kg diagnostic
    # -------------------------------------------------

    q850 = get_level(q, 850)
    moisture_850 = q850 * 1000.0

    # Coordinates
    latitude = t500_c["latitude"].values
    longitude = t500_c["longitude"].values

    # Forecast times / steps are retained from ECMWF
    if "step" in t500_c.coords:
        steps = t500_c["step"].values
    else:
        steps = np.arange(t500_c.shape[0])

    OUTPUT_FILE.parent.mkdir(exist_ok=True)

    np.savez_compressed(
        OUTPUT_FILE,
        latitude=latitude,
        longitude=longitude,
        steps=steps,
        t500_c=t500_c.values,
        lapse_700_500=lapse_700_500.values,
        shear_850_300=shear_850_300.values,
        omega_700=omega_700.values,

omega_500=omega_500.values,
        moisture_850=moisture_850.values,
    )

    print("Real ECMWF hail diagnostics created:")
    print(OUTPUT_FILE)

    print(
        "T500 range:",
        float(np.nanmin(t500_c.values)),
        float(np.nanmax(t500_c.values)),
    )

    print(
        "700-500 lapse-rate range:",
        float(np.nanmin(lapse_700_500.values)),
        float(np.nanmax(lapse_700_500.values)),
    )

    print(
        "850-300 wind-difference range:",
        float(np.nanmin(shear_850_300.values)),
        float(np.nanmax(shear_850_300.values)),
    )
print(
    "Omega 700 range:",
    float(np.nanmin(omega_700.values)),
    float(np.nanmax(omega_700.values)),
)

print(
    "Omega 500 range:",
    float(np.nanmin(omega_500.values)),
    float(np.nanmax(omega_500.values)),
)

if __name__ == "__main__":
    main()
