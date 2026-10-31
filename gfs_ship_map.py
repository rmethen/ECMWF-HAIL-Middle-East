"""Experimental GFS SHIP numerical calculation, independent of hail V2.

Reference: SHARPpy sharptab.params.ship (SPC-corrected formulation).
https://github.com/sharppy/SHARPpy/blob/main/sharppy/sharptab/params.py

This is a calculation kernel only. Never publish an image without actual,
same-valid-time GFS most-unstable parcel, freezing level AGL and shear data.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ShipInputs:
    mucape_jkg: np.ndarray
    mu_mixing_ratio_gkg: np.ndarray
    lapse_700_500_k_km: np.ndarray
    temperature_500_c: np.ndarray
    bulk_shear_0_6km_ms: np.ndarray
    freezing_level_agl_m: np.ndarray


def validate_inputs(fields: ShipInputs) -> tuple[int, ...]:
    values = [np.asarray(getattr(fields, key), dtype=float)
              for key in fields.__dataclass_fields__]
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("SHIP input grids have inconsistent shapes")
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("SHIP inputs contain missing/non-finite data")
    cape, mix, lapse, t500, shear, freezing = values
    if np.any(cape < 0) or np.any(mix < 0) or np.any(shear < 0) or np.any(freezing < 0):
        raise ValueError("Negative CAPE, mixing ratio, shear or freezing level")
    if np.any((mix > 40) | (shear > 150) | (t500 < -110) | (t500 > 40)
              | (lapse < -20) | (lapse > 30) | (freezing > 15000)):
        raise ValueError("Out-of-range SHIP inputs: check units")
    return shape


def compute_ship(fields: ShipInputs) -> np.ndarray:
    """Compute SPC-style SHIP (dimensionless), with SHARPpy corrections.

    Note: 0-6km shear is vector bulk shear, not 850-300 hPa shear.
    Freezing level is AGL, not MSL. Inputs must be actual MU parcel fields.
    """
    validate_inputs(fields)
    cape = np.asarray(fields.mucape_jkg, dtype=float)
    mix = np.clip(fields.mu_mixing_ratio_gkg, 11.0, 13.6)
    lapse = np.asarray(fields.lapse_700_500_k_km, dtype=float)
    t500 = np.minimum(fields.temperature_500_c, -5.5)
    shear = np.clip(fields.bulk_shear_0_6km_ms, 7.0, 27.0)
    freezing = np.asarray(fields.freezing_level_agl_m, dtype=float)
    score = cape * mix * lapse * (-t500) * shear / 42_000_000.0
    score *= np.minimum(cape / 1300.0, 1.0)
    score *= np.minimum(np.maximum(lapse, 0.0) / 5.8, 1.0)
    score *= np.minimum(freezing / 2400.0, 1.0)
    return np.maximum(score, 0.0)
