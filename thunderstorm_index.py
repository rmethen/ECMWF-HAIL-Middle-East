"""CAPE-free experimental thunderstorm and lightning potential diagnostic."""

import numpy as np

from hail_index import inverse_scale, scale


def thunderstorm_potential(li850, omega700, omega500, moisture_850, lapse_700_500,
                           shear_850_300, t500_c):
    """Return a 0–100 potential index; this is not forecast flash density."""
    instability = inverse_scale(li850, -7.0, 2.0)
    ascent700 = inverse_scale(omega700, -1.2, 0.15)
    ascent500 = inverse_scale(omega500, -0.9, 0.12)
    ascent = 0.55 * ascent700 + 0.45 * ascent500
    moisture = scale(moisture_850, 4.0, 13.0)
    lapse = scale(lapse_700_500, 5.0, 8.0)
    shear = scale(shear_850_300, 6.0, 28.0)
    cold_midlevels = inverse_scale(t500_c, -24.0, -8.0)

    score = (
        0.25 * instability
        + 0.25 * ascent
        + 0.20 * moisture
        + 0.15 * lapse
        + 0.10 * shear
        + 0.05 * cold_midlevels
    )
    score *= 0.35 + 0.65 * moisture
    score *= 0.40 + 0.60 * np.maximum(instability, ascent)
    return np.clip(score * 100.0, 0.0, 100.0)
