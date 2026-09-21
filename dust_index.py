"""Synoptic dust-storm and wall-dust diagnostic."""

import numpy as np
from hail_index import inverse_scale, scale


def dust_storm_potential(gust10, msl_hpa, jet300_kmh, jet200_kmh,
                         jet_interaction, cold_front_gradient, polar_low):
    gust = scale(gust10, 12.0, 30.0)
    gale = scale(gust10, 17.22, 25.0)
    if np.ndim(gust10) >= 3:
        gale_hits = np.sum(np.asarray(gust10) >= 17.22, axis=0)
        gale_persistence_2d = scale(gale_hits, 1.0, 3.0)
        gale_persistence = np.broadcast_to(
            gale_persistence_2d, np.asarray(gust10).shape
        )
    else:
        gale_persistence = (np.asarray(gust10) >= 17.22).astype(float)
    deep_low = inverse_scale(msl_hpa, 985.0, 1005.0)
    polar_jet = scale(jet300_kmh, 90.0, 160.0)
    subtropical_jet = scale(jet200_kmh, 90.0, 170.0)
    jet_coupling = np.clip(jet_interaction, 0.0, 1.0)
    cold_front = scale(cold_front_gradient, 1.0, 5.0)
    polar_low_score = np.clip(polar_low, 0.0, 1.0)

    synoptic = (
        0.32 * gust + 0.23 * deep_low + 0.18 * polar_jet
        + 0.12 * subtropical_jet + 0.10 * cold_front
        + 0.05 * polar_low_score
    )
    frontal_wall = (
        0.27 * gust + 0.20 * gale + 0.15 * gale_persistence
        + 0.15 * deep_low + 0.10 * polar_jet
        + 0.08 * jet_coupling + 0.05 * cold_front
    )
    polar_wall = (
        0.28 * gust + 0.20 * gale + 0.18 * deep_low
        + 0.14 * polar_low_score + 0.12 * cold_front
        + 0.08 * polar_jet
    )
    wall = np.maximum(frontal_wall, polar_wall)
    wall *= 0.85 + 0.25 * gale_persistence
    combined = np.maximum(synoptic, wall)
    return (
        np.clip(combined * 100.0, 0.0, 100.0),
        np.clip(wall * 100.0, 0.0, 100.0),
        np.clip(gale_persistence * 100.0, 0.0, 100.0),
    )
