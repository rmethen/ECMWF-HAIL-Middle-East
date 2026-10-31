"""Unit tests for independent GFS SHIP GRIB request builder."""
import unittest
from datetime import datetime, timezone, timedelta
from urllib.parse import urlsplit, parse_qs
from gfs_ship_inputs import pressure_profile_url, PRESSURE_LEVELS, input_status


class ShipRequestTests(unittest.TestCase):
    def setUp(self):
        self.cycle = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)

    def test_url_has_required_fields_and_levels(self):
        query = parse_qs(urlsplit(pressure_profile_url(self.cycle, 6)).query)
        self.assertEqual(query["file"], ["gfs.t12z.pgrb2.0p25.f006"])
        self.assertEqual(query["var_SPFH"], ["on"])
        self.assertEqual(query["var_HGT"], ["on"])
        for level in PRESSURE_LEVELS:
            self.assertEqual(query[f"lev_{level}_mb"], ["on"])

    def test_bad_cycle_or_hour_rejected(self):
        for cycle in (self.cycle.replace(tzinfo=None),
                      self.cycle.astimezone(timezone(timedelta(hours=3))),
                      self.cycle.replace(hour=11)):
            with self.assertRaises(ValueError):
                pressure_profile_url(cycle, 6)
        for hour in (-6, 3, 78):
            with self.assertRaises(ValueError):
                pressure_profile_url(self.cycle, hour)

    def test_publication_remains_blocked(self):
        self.assertEqual(input_status()["image_publication"], "blocked")


if __name__ == "__main__":
    unittest.main()
