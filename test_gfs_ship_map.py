"""Safety tests for the independent GFS SHIP development scaffold."""
import unittest
import numpy as np
from gfs_ship_map import ShipInputs, validate_inputs, compute_ship


def sample():
    x = lambda value: np.full((2, 3), value, dtype=float)
    return ShipInputs(x(1200), x(12), x(7), x(-15), x(20))


class ShipValidationTests(unittest.TestCase):
    def test_complete_fields(self):
        self.assertEqual(validate_inputs(sample()), (2, 3))

    def test_mismatched_grid_rejected(self):
        s = sample()
        with self.assertRaises(ValueError):
            validate_inputs(ShipInputs(s.mucape_jkg, np.ones((3, 3)),
                                       s.lapse_700_500_k_km,
                                       s.temperature_500_c,
                                       s.bulk_shear_0_6km_ms))

    def test_missing_field_rejected(self):
        s = sample()
        s.mucape_jkg[0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_inputs(s)

    def test_unvalidated_product_not_published(self):
        with self.assertRaises(NotImplementedError):
            compute_ship(sample())


if __name__ == "__main__":
    unittest.main()
