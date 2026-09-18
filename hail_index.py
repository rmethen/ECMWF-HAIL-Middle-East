"""
ECMWF Experimental Hail Potential Index V2
Middle East

Experimental diagnostic — not an official ECMWF product.
"""

import numpy as np


def scale(x, low, high):
    """Normalize a field to 0–1."""
    x = np.asarray(x, dtype=float)
    return np.clip((x - low) / (high - low), 0.0, 1.0)


def inverse_scale(x, low, high):
    """Higher score for lower values."""
    x = np.asarray(x, dtype=float)
    return np.clip((high - x) / (high - low), 0.0, 1.0)


def hail_potential_v2(
    mucape,
    shear06,
    wbz_m,
    hgl_depth_m,
    lapse_700_500,
    t500_c,
    mixing_ratio,
):
    """
    Calculate experimental hail potential from 0–100.

    Inputs:
      mucape          J/kg
      shear06         m/s
      wbz_m           wet-bulb-zero height, metres
      hgl_depth_m     -10C to -30C hail-growth-layer depth, metres
      lapse_700_500   C/km
      t500_c          500-hPa temperature, C
      mixing_ratio    g/kg
    """

    # Hail-growth environment
    hgl = scale(hgl_depth_m, 2000.0, 5000.0)

    # Moderate WBZ generally favors hail survival.
    wbz_low = scale(wbz_m, 1200.0, 2200.0)
    wbz_high = inverse_scale(wbz_m, 3200.0, 4500.0)
    wbz = np.minimum(wbz_low, wbz_high)

    # Deep-layer shear
    shear = scale(shear06, 8.0, 25.0)

    # Instability deliberately limited so CAPE does not dominate.
    instability = scale(mucape, 250.0, 2000.0)

    # Mid-level lapse rate
    lapse = scale(lapse_700_500, 5.5, 8.0)

    # Cold mid-level temperatures
    t500 = inverse_scale(t500_c, -22.0, -10.0)

    # Low-level moisture
    moisture = scale(mixing_ratio, 5.0, 13.0)

    # V2 weighting
    score = (
        0.25 * hgl
        + 0.20 * wbz
        + 0.20 * shear
        + 0.12 * instability
        + 0.10 * lapse
        + 0.07 * t500
        + 0.06 * moisture
    )

    return np.clip(score * 100.0, 0.0, 100.0)


def category(index):
    """Simple descriptive categories."""
    index = np.asarray(index)

    return np.select(
        [
            index < 20,
            index < 40,
            index < 60,
            index < 80,
            index >= 80,
        ],
        [
            "Very Low",
            "Low",
            "Moderate",
            "High",
            "Very High",
        ],
        default="Unknown",
    )


if __name__ == "__main__":
    print("ECMWF Experimental Hail Potential Index V2")
    print("Middle East")
    print("Core calculation module loaded successfully.")
