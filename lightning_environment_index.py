"""Independent, experimental ECMWF lightning-environment diagnostic.

The score describes ingredients on the model grid, not flash probability or
lightning density. In particular MUCAPE alone cannot overcome weak lift or a
dry lower troposphere.
"""

import numpy as np

from hail_index import inverse_scale, scale


def lightning_environment_potential(mucape, li850, omega700, omega500,
                                    moisture_850, lapse_700_500,
                                    total_totals, kuwait_total_totals):
    """Return a 0–100 exploratory score for arrays of the same shape."""
    cape = scale(mucape, 250.0, 2200.0)
    li = inverse_scale(li850, -7.0, 2.0)
    lift = (0.6 * inverse_scale(omega700, -1.0, 0.1)
            + 0.4 * inverse_scale(omega500, -0.8, 0.1))
    moisture = scale(moisture_850, 4.0, 13.0)
    lapse = scale(lapse_700_500, 5.0, 8.0)
    tti = (0.5 * scale(total_totals, 42.0, 58.0)
           + 0.5 * scale(kuwait_total_totals, 42.0, 62.0))
    ingredients = (0.18 * cape + 0.23 * li + 0.17 * tti
                   + 0.20 * lift + 0.13 * moisture + 0.09 * lapse)
    # A favorable CAPE/TTI field without moisture and a lifting mechanism is
    # a weak lightning signal. Keep low background values for comparison.
    support = np.minimum(moisture, np.maximum(lift, 0.5 * li))
    return np.clip(100.0 * ingredients * (0.25 + 0.75 * support), 0.0, 100.0)
