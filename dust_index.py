"""Experimental dust-storm and convective-wall-dust diagnostics."""

import numpy as np
from hail_index import inverse_scale, scale


def dust_storm_potential(wind10, gust10, msl_hpa, dewpoint_depression,
                         thunderstorm, omega700):
    wind = scale(wind10, 8.0, 22.0)
    gust = scale(gust10, 12.0, 30.0)
    # Xasteria Wind Warning: large circle = gale or above (>62 km/h).
    # ECMWF winds are m/s, so 62 km/h = 17.22 m/s.
    gale = scale(gust10, 17.22, 25.0)
    if np.ndim(gust10) >= 3:
        gale_hits = np.sum(np.asarray(gust10) >= 17.22, axis=0)
        gale_persistence_2d = scale(gale_hits, 1.0, 3.0)
        gale_persistence = np.broadcast_to(
            gale_persistence_2d, np.asarray(gust10).shape
        )
    else:
        gale_persistence = (np.asarray(gust10) >= 17.22).astype(float)
    deep_low = inverse_scale(msl_hpa, 985.0, 1012.0)
    dry_surface = scale(dewpoint_depression, 4.0, 20.0)
    convective = scale(thunderstorm, 25.0, 80.0)
    downdraft_proxy = inverse_scale(omega700, -1.0, 0.20)

    synoptic = (
        0.30 * gust + 0.20 * gale + 0.20 * wind
        + 0.15 * deep_low + 0.15 * dry_surface
    )
    wall = (
        0.32 * gust + 0.25 * gale + 0.23 * convective
        + 0.12 * downdraft_proxy + 0.08 * dry_surface
    )
    # Repeated large circles strengthen confidence. A wall/haboob still
    # requires convective/outflow support, not gale wind alone.
    wall *= 0.35 + 0.65 * np.minimum(1.0, np.maximum(convective, gale))
    wall *= 0.85 + 0.25 * gale_persistence
    combined = np.maximum(synoptic, wall)
    return (
        np.clip(combined * 100.0, 0.0, 100.0),
        np.clip(wall * 100.0, 0.0, 100.0),
        np.clip(gale_persistence * 100.0, 0.0, 100.0),
    )
