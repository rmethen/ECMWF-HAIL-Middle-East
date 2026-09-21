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
    shear06,
    wbz_m,
    hgl_depth_m,
    lapse_700_500,
    t500_c,
    mixing_ratio,
    li850,
    omega700,
    omega500,
):
    """
    Calculate an experimental large-hail-environment index from 0–100.

    Inputs:
      shear06         m/s
      wbz_m           wet-bulb-zero height, metres
      hgl_depth_m     -10C to -30C hail-growth-layer depth, metres
      lapse_700_500   C/km
      t500_c          500-hPa temperature, C
      mixing_ratio    g/kg
      li850           C
      omega700        Pa/s (negative values indicate ascent)
      omega500        Pa/s (negative values indicate ascent)
    """

    # A deep hail-growth layer is more relevant to large-hail support than to
    # the occurrence of small hail alone.
    hgl = scale(hgl_depth_m, 2400.0, 5000.0)

    # Moderate WBZ generally favors hail survival.
    wbz_low = scale(wbz_m, 1400.0, 2200.0)
    wbz_high = inverse_scale(wbz_m, 3300.0, 4400.0)
    wbz = np.minimum(wbz_low, wbz_high)

    # Deep-layer shear
    shear = scale(shear06, 10.0, 28.0)

    # Mid-level lapse rate
    lapse = scale(lapse_700_500, 5.5, 8.0)

    # Cold mid-level temperatures
    t500 = inverse_scale(t500_c, -22.0, -10.0)

    # Low-level moisture
    moisture = scale(mixing_ratio, 5.0, 13.0)
    li = inverse_scale(li850, -6.0, 2.0)
    ascent700 = inverse_scale(omega700, -1.0, 0.2)
    ascent500 = inverse_scale(omega500, -0.8, 0.15)
    ascent = 0.60 * ascent700 + 0.40 * ascent500
    # Hail physics dominate the base score.  Convective gates below prevent
    # broad false-positive shading where HGL/WBZ are favourable but storms
    # are unlikely to develop.
    score = (
        0.32 * hgl
        + 0.25 * wbz
        + 0.20 * shear
        + 0.09 * lapse
        + 0.06 * t500
        + 0.03 * moisture
        + 0.025 * li
        + 0.025 * ascent
    )
    convective_support = np.maximum(li, ascent)
    score *= 0.45 + 0.55 * convective_support
    score *= 0.70 + 0.30 * moisture
    # Large-hail signal requires both an efficient growth layer and storm
    # organisation. This suppresses ordinary small-hail environments.
    score *= 0.45 + 0.55 * np.minimum(hgl, shear)

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
    print("ECMWF Experimental Large Hail Potential Index V3")
    print("Middle East")
    print("Core calculation module loaded successfully.")
