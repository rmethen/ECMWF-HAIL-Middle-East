"""Process ECMWF pressure-level fields for the experimental indices."""

from pathlib import Path

import numpy as np
import xarray as xr

GRIB_FILE = Path("data/ecmwf_hail_0_72h.grib2")
OUTPUT_FILE = Path("data/hail_diagnostics.npz")


def open_field(short_name):
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


def level_coord(da):
    for name in ("isobaricInhPa", "level"):
        if name in da.coords:
            return name
    raise KeyError("Pressure-level coordinate not found")


def get_level(da, level):
    return da.sel({level_coord(da): level})


def height_at_isotherm(temp_c, height_m, target_c):
    """Linearly interpolate the first height where temperature crosses target."""
    axis = temp_c.get_axis_num(level_coord(temp_c))
    temperature = np.moveaxis(np.asarray(temp_c), axis, 0)
    height = np.moveaxis(np.asarray(height_m), axis, 0)
    result = np.full(temperature.shape[1:], np.inf, dtype=float)

    for k in range(temperature.shape[0] - 1):
        t0, t1 = temperature[k], temperature[k + 1]
        z0, z1 = height[k], height[k + 1]
        delta = t1 - t0
        crossing = (
            np.isfinite(t0)
            & np.isfinite(t1)
            & np.isfinite(z0)
            & np.isfinite(z1)
            & ((t0 - target_c) * (t1 - target_c) <= 0.0)
            & (np.abs(delta) > 1.0e-6)
        )
        fraction = (target_c - t0) / np.where(
            np.abs(delta) > 1.0e-6, delta, np.nan
        )
        candidate = z0 + fraction * (z1 - z0)
        result = np.where(crossing, np.minimum(result, candidate), result)

    return np.where(np.isfinite(result), result, np.nan)


def approximate_wet_bulb_c(temp_c, specific_humidity, pressure_hpa):
    """Stull wet-bulb approximation from T, q and pressure."""
    axis = temp_c.get_axis_num(level_coord(temp_c))
    t = np.asarray(temp_c, dtype=float)
    q = np.asarray(specific_humidity, dtype=float)
    shape = [1] * t.ndim
    shape[axis] = pressure_hpa.size
    p = np.asarray(pressure_hpa, dtype=float).reshape(shape)
    vapour_pressure = q * p / (0.622 + 0.378 * q)
    saturation = 6.112 * np.exp(17.67 * t / (t + 243.5))
    rh = np.clip(100.0 * vapour_pressure / saturation, 1.0, 100.0)
    return (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def approximate_li850(t850_k, td850_k, t500_k, z850_m, z500_m):
    """Vectorized 850-to-500 hPa lifted-index approximation."""
    t850_k = np.asarray(t850_k, dtype=float)
    td850_k = np.asarray(td850_k, dtype=float)
    td850_c = td850_k - 273.15
    t850_c = t850_k - 273.15
    lcl_temp_k = 1.0 / (
        1.0 / np.maximum(td850_k - 56.0, 1.0)
        + np.log(t850_k / np.maximum(td850_k, 150.0)) / 800.0
    ) + 56.0
    lcl_height_m = np.asarray(z850_m) + 125.0 * np.maximum(t850_c - td850_c, 0.0)
    parcel500_k = lcl_temp_k - 6.0 * np.maximum(
        np.asarray(z500_m) - lcl_height_m, 0.0
    ) / 1000.0
    dry500_k = t850_k * (500.0 / 850.0) ** 0.2854
    lcl_pressure = 850.0 * (lcl_temp_k / t850_k) ** (1.0 / 0.2854)
    parcel500_k = np.where(lcl_pressure < 500.0, dry500_k, parcel500_k)
    return np.asarray(t500_k) - parcel500_k


def main():
    if not GRIB_FILE.exists():
        raise FileNotFoundError(f"Missing ECMWF GRIB file: {GRIB_FILE}")

    print("Opening real ECMWF GRIB2 fields...")
    t = open_field("t")["t"]
    u = open_field("u")["u"]
    v = open_field("v")["v"]
    q = open_field("q")["q"]
    gh = open_field("gh")["gh"]
    w = open_field("w")["w"]

    t_c = t - 273.15
    t500_c = get_level(t_c, 500)
    t700_c = get_level(t_c, 700)
    z500 = get_level(gh, 500)
    z700 = get_level(gh, 700)
    depth_km = np.maximum((z500 - z700) / 1000.0, 0.1)
    lapse_700_500 = (t700_c - t500_c) / depth_km

    u850, v850 = get_level(u, 850), get_level(v, 850)
    u300, v300 = get_level(u, 300), get_level(v, 300)
    shear_850_300 = np.hypot(u300 - u850, v300 - v850)
    omega_700 = get_level(w, 700)
    omega_500 = get_level(w, 500)

    q850_values = get_level(q, 850).values
    moisture_850 = q850_values * 1000.0
    t850_values = get_level(t, 850).values
    vapour_pressure_850 = q850_values * 850.0 / (0.622 + 0.378 * q850_values)
    log_ratio = np.log(np.maximum(vapour_pressure_850, 0.01) / 6.112)
    td850_k = 243.5 * log_ratio / (17.67 - log_ratio) + 273.15
    li850 = approximate_li850(
        t850_values, td850_k, get_level(t, 500).values,
        get_level(gh, 850).values, get_level(gh, 500).values,
    )

    level_name = level_coord(t_c)
    pressures = t_c[level_name].values
    wet_bulb_c = approximate_wet_bulb_c(t_c, q, pressures)
    wet_bulb_da = xr.DataArray(wet_bulb_c, coords=t_c.coords, dims=t_c.dims)
    wbz_m = height_at_isotherm(wet_bulb_da, gh, 0.0)
    h_minus10 = height_at_isotherm(t_c, gh, -10.0)
    h_minus30 = height_at_isotherm(t_c, gh, -30.0)
    hgl_depth_m = np.maximum(h_minus30 - h_minus10, 0.0)

    latitude = t500_c["latitude"].values
    longitude = t500_c["longitude"].values
    steps = (
        t500_c["step"].values
        if "step" in t500_c.coords
        else np.arange(t500_c.shape[0])
    )
    init_time = (
        t500_c["time"].values
        if "time" in t500_c.coords
        else np.datetime64("NaT")
    )

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        latitude=latitude,
        longitude=longitude,
        steps=steps,
        init_time=init_time,
        t500_c=t500_c.values,
        lapse_700_500=lapse_700_500.values,
        shear_850_300=shear_850_300.values,
        omega_700=omega_700.values,
        omega_500=omega_500.values,
        z500_m=z500.values,
        moisture_850=moisture_850,
        li850=li850,
        wbz_m=wbz_m,
        hgl_depth_m=hgl_depth_m,
    )
    print("Real ECMWF diagnostics created:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
