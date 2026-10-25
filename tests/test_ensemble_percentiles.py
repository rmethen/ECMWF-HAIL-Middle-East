"""Check pointwise ensemble quantiles against hand-calculated amounts."""

import unittest

import numpy as np

from ensemble_percentiles import precipitation_percentiles


class PrecipitationPercentileTests(unittest.TestCase):
    def test_quantiles_are_pointwise_and_include_dry_members(self):
        rain = np.zeros((5, 2, 2))
        rain[:, 0, 0] = [0, 10, 20, 30, 100]
        rain[:, 0, 1] = [100, 30, 20, 10, 0]
        rain[:, 1, 0] = [0, 0, 0, 0, 100]
        p80, p90 = precipitation_percentiles(rain)
        np.testing.assert_allclose(p80, [[44, 44], [20, 0]])
        np.testing.assert_allclose(p90, [[72, 72], [60, 0]])
        np.testing.assert_array_less(p80, p90 + 1e-8)
        # Reordering members must not change the composite quantile map.
        np.testing.assert_allclose(
            precipitation_percentiles(rain[::-1]), [p80, p90]
        )

    def test_reject_incomplete_ensemble(self):
        for value in (np.nan, np.inf, -1):
            rain = np.zeros((5, 2, 2))
            rain[0, 0, 0] = value
            with self.assertRaises(ValueError):
                precipitation_percentiles(rain)
        with self.assertRaises(ValueError):
            precipitation_percentiles(np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
