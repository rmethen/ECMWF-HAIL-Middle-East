"""Numerical regression and safety checks for SHARPpy-style SHIP."""
import unittest
import numpy as np
from gfs_ship_map import ShipInputs, validate_inputs, compute_ship


def sample(cape=2000., mixing=12., lapse=7., t500=-15., shear=20., freezing=3000.):
    x = lambda value: np.full((2, 3), value, dtype=float)
    return ShipInputs(x(cape), x(mixing), x(lapse), x(t500), x(shear), x(freezing))


class ShipTests(unittest.TestCase):
    def test_complete_fields(self):
        self.assertEqual(validate_inputs(sample()), (2, 3))

    def test_formula_reference_case(self):
        expected = 2000 * 12 * 7 * 15 * 20 / 42_000_000
        np.testing.assert_allclose(compute_ship(sample()), expected)

    def test_low_cape_lapse_and_freezing_adjustments(self):
        s = sample(cape=650, lapse=2.9, freezing=1200)
        expected = (650 * 12 * 2.9 * 15 * 20 / 42_000_000
                    * (650 / 1300) * (2.9 / 5.8) * (1200 / 2400))
        np.testing.assert_allclose(compute_ship(s), expected)

    def test_shear_mixing_and_warm_t500_caps(self):
        s = sample(mixing=20, t500=-2, shear=50)
        expected = 2000 * 13.6 * 7 * 5.5 * 27 / 42_000_000
        np.testing.assert_allclose(compute_ship(s), expected)

    def test_missing_and_mismatched_grid_rejected(self):
        s = sample()
        s.mucape_jkg[0, 0] = np.nan
        with self.assertRaises(ValueError):
            compute_ship(s)
        s = sample()
        with self.assertRaises(ValueError):
            compute_ship(ShipInputs(s.mucape_jkg, np.ones((3, 3)),
                                    s.lapse_700_500_k_km, s.temperature_500_c,
                                    s.bulk_shear_0_6km_ms, s.freezing_level_agl_m))

    def test_negative_inputs_rejected(self):
        with self.assertRaises(ValueError):
            compute_ship(sample(freezing=-100))


if __name__ == "__main__":
    unittest.main()
