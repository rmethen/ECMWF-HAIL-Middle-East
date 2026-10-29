"""GFS SHIP development scaffold.

This module intentionally does not publish a SHIP image until all required
meteorological inputs and the operational formulation are validated.
Do not substitute the existing GFS V2 hail score for SHIP.
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


def validate_inputs(fields: ShipInputs) -> tuple[int, ...]:
    """Reject missing, inconsistent or implausible fields before calculation."""
    values = [np.asarray(getattr(fields, key), dtype=float) for key in
              fields.__dataclass_fields__]
    shape = values[0].shape
    if any(value.shape != shape for value in values):
        raise ValueError("SHIP input grids have inconsistent shapes")
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("SHIP inputs contain missing/non-finite data")
    cape, mix, lapse, t500, shear = values
    if np.any(cape < 0) or np.any(mix < 0) or np.any(shear < 0):
        raise ValueError("Negative CAPE, mixing ratio or shear")
    if np.any((mix > 40) | (shear > 150) | (t500 < -110) | (t500 > 40)
              | (lapse < -20) | (lapse > 30)):
        raise ValueError("Out-of-range SHIP inputs: check units")
    return shape


def compute_ship(fields: ShipInputs):
    """Fail closed until the published SPC equation and conventions are tested.

    Requirements: most-unstable parcel CAPE and mixing ratio, 700–500 hPa
    lapse rate, 500-hPa temperature, 0–6 km bulk shear; all from same GFS
    valid time. Operational thresholds, caps and units must be independently
    verified before this function is enabled.
    """
    validate_inputs(fields)
    raise NotImplementedError(
        "SHIP publication blocked pending formula, parcel and unit validation"
    )
