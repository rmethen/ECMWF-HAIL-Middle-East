"""Experimental dust-storm and convective-wall-dust diagnostics."""

import numpy as np
from hail_index import inverse_scale, scale


def dust_storm_potential(wind10, gust10, msl_hpa, dewpoint_depression,
                         thunderstorm, omega700):
    wind = scale(wind10, 8.0, 22.0)
    gust = scale(gust10, 12.0, 30.0)
    deep_low = inverse_scale(msl_hpa, 985.0, 1012.0)
    dry_surface = scale(dewpoint_depression, 4.0, 20.0)
    convective = scale(thunderstorm, 25.0, 80.0)
    downdraft_proxy = inverse_scale(omega700, -1.0, 0.20)

    synoptic = (0.38 * gust + 0.27 * wind + 0.20 * deep_low + 0.15 * dry_surface)
    wall = (0.42 * gust + 0.28 * convective + 0.18 * downdraft_proxy + 0.12 * dry_surface)
    # Wall-dust needs both convective forcing and strong outflow.
    wall *= 0.35 + 0.65 * np.minimum(1.0, np.maximum(convective, gust))
    combined = np.maximum(synoptic, wall)
    return np.clip(combined * 100.0, 0.0, 100.0), np.clip(wall * 100.0, 0.0, 100.0)
