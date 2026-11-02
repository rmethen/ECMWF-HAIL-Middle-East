"""Offline numerical and request-contract tests; no NOAA network needed."""
import unittest
from unittest.mock import call, patch
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import numpy as np

from gfs_upper_dynamics import (
    EARTH_RADIUS_M, EARTH_ROTATION_RAD_S,
    _vector_derivatives, build_url, calculate_diagnostics, run_hours,
    find_ascent_cores,
)


class GFSUpperDynamicsTests(unittest.TestCase):
    def setUp(self):
        self.lon = np.arange(43., 53. + 0.001, 1.)
        self.lat = np.arange(25., 35. + 0.001, 1.)
        self.shape = (len(self.lat), len(self.lon))
        self.zero = np.zeros(self.shape)

    def test_url_has_real_noaa_pressure_levels_and_bounded_region(self):
        c = datetime(2026, 11, 1, 18, tzinfo=timezone.utc)
        parsed = urlsplit(build_url(c, 6))
        params = parse_qs(parsed.query, keep_blank_values=True)
        self.assertEqual(parsed.hostname, "nomads.ncep.noaa.gov")
        self.assertEqual(params["file"], ["gfs.t18z.pgrb2.0p25.f006"])
        self.assertEqual(params["dir"], ["/gfs.20261101/18/atmos"])
        for variable in ("UGRD", "VGRD", "VVEL"):
            self.assertEqual(params[f"var_{variable}"], ["on"])
        for lev in (250, 400, 500, 600):
            self.assertEqual(params[f"lev_{lev}_mb"], ["on"])
        self.assertEqual(params["leftlon"], ["30"])
        self.assertEqual(params["rightlon"], ["68"])

    def test_invalid_cycle(self):
        from datetime import timedelta
        with self.assertRaises(ValueError):
            build_url(datetime(2026, 11, 1, 19, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            build_url(datetime(2026, 11, 1, 18))
        with self.assertRaises(ValueError):
            build_url(datetime(2026, 11, 1, 18, tzinfo=timezone(timedelta(hours=3))))

    def test_zero_wind_has_zero_divergence_and_relative_vorticity(self):
        div, vort = _vector_derivatives(self.lon, self.lat, self.zero, self.zero)
        np.testing.assert_allclose(div, 0., atol=1e-14)
        np.testing.assert_allclose(vort, 0., atol=1e-14)

    def test_linear_zonal_wind_gives_expected_divergence(self):
        strength = 2e-5  # s^-1
        u = EARTH_RADIUS_M * np.cos(np.deg2rad(self.lat))[:, None] * (
            np.deg2rad(self.lon) - np.deg2rad(48.)) * strength
        u = np.broadcast_to(u, self.shape)
        div, _ = _vector_derivatives(self.lon, self.lat, u, self.zero)
        np.testing.assert_allclose(div[:, 1:-1], strength, rtol=1e-11, atol=1e-12)

    def test_descending_latitudes_are_supported(self):
        z = np.zeros(self.shape)
        up = calculate_diagnostics(
            self.lon, self.lat, z, z, z, z, z-.3, z-.6, z-.9)
        down = calculate_diagnostics(
            self.lon, self.lat[::-1], z[::-1], z[::-1],
            z[::-1], z[::-1], (z-.3)[::-1], (z-.6)[::-1], (z-.9)[::-1])
        for k in up:
            np.testing.assert_allclose(up[k], down[k][::-1], atol=1e-11)

    def test_ascent_sign_and_planetary_vorticity(self):
        z = self.zero
        fields = calculate_diagnostics(
            self.lon, self.lat, z, z, z, z, z-.3, z-.6, z-.9)
        np.testing.assert_allclose(fields["upward_omega_pa_s"], .6, atol=1e-13)
        expected = 2 * EARTH_ROTATION_RAD_S * np.sin(
            np.deg2rad(self.lat))[:, None] * 1e5
        np.testing.assert_allclose(fields["rel_vort500_1e5_s"], 0., atol=1e-13)
        np.testing.assert_allclose(fields["abs_vort500_1e5_s"], np.broadcast_to(expected, self.shape))

    def test_invalid_shapes_rejected(self):
        z = self.zero
        with self.assertRaises(ValueError):
            calculate_diagnostics(self.lon, self.lat, z, z, z, z, z, z, z[:, 1:])


    def test_multi_hour_series_shares_initialization(self):
        cycle = datetime(2026, 11, 1, 18, tzinfo=timezone.utc)
        with patch("gfs_upper_dynamics.download", side_effect=lambda c, h, path: path) as get_data, \
             patch("gfs_upper_dynamics.read_diagnostics",
                   return_value=(self.lon, self.lat, {"ok": self.zero})), \
             patch("gfs_upper_dynamics.render", side_effect=lambda lon, lat, diag, c, h, dest: dest) as draw:
            paths = run_hours(cycle=cycle, hours=(3, 6, 12))
        self.assertEqual(len(paths), 3)
        self.assertEqual(get_data.call_count, 3)
        self.assertEqual(draw.call_count, 3)
        self.assertTrue(all(args.args[0] == cycle for args in get_data.call_args_list))
        self.assertTrue(str(paths[0]).endswith("_F003.png"))
        self.assertTrue(str(paths[1]).endswith("_F006.png"))
        self.assertTrue(str(paths[2]).endswith("_F012.png"))

    def test_multi_hour_rejects_duplicate_or_excessive_leads(self):
        c = datetime(2026, 11, 1, 18, tzinfo=timezone.utc)
        for hours in ((), (6, 6), (-1,), (121,), tuple(range(6))):
            with self.subTest(hours=hours), self.assertRaises(ValueError):
                run_hours(cycle=c, hours=hours)


    def test_strong_ascent_cores_and_subthreshold_rejection(self):
        ascent = np.zeros(self.shape)
        ascent[2, 2] = 0.32
        ascent[7, 7] = 0.47
        ascent[4, 9] = 0.23
        cores = find_ascent_cores(ascent, threshold=0.25, neighborhood=5)
        self.assertTrue(cores[2, 2])
        self.assertTrue(cores[7, 7])
        self.assertFalse(cores[4, 9])
        self.assertEqual(int(cores.sum()), 2)

    def test_adjacent_weaker_peak_not_selected(self):
        ascent = np.zeros(self.shape)
        ascent[5, 5] = 0.50
        ascent[5, 6] = 0.42
        cores = find_ascent_cores(ascent, threshold=0.25, neighborhood=5)
        self.assertTrue(cores[5, 5])
        self.assertFalse(cores[5, 6])
        self.assertEqual(int(cores.sum()), 1)

    def test_plateau_produces_only_one_dot(self):
        ascent = np.zeros(self.shape)
        ascent[4:6, 4:6] = 0.40
        cores = find_ascent_cores(ascent, threshold=0.25, neighborhood=5)
        self.assertEqual(int(cores.sum()), 1)
        self.assertTrue(cores[4:6, 4:6].any())

    def test_default_albany_scale_half_pascal_and_wider_suppression(self):
        ascent = np.zeros(self.shape)
        ascent[2, 2] = 0.49  # below the 0.5 Pa/s plotting threshold
        ascent[7, 7] = 0.72
        ascent[7, 9] = 0.64  # weaker neighbor inside default 9x9 window
        cores = find_ascent_cores(ascent)
        self.assertFalse(cores[2, 2])
        self.assertTrue(cores[7, 7])
        self.assertFalse(cores[7, 9])
        self.assertEqual(int(cores.sum()), 1)

    def test_ascent_cores_handle_nan_and_reject_invalid_inputs(self):
        ascent = np.zeros(self.shape)
        ascent[1, 1] = np.nan
        ascent[7, 7] = 0.36
        cores = find_ascent_cores(ascent, threshold=0.25, neighborhood=5)
        self.assertTrue(cores[7, 7])
        self.assertFalse(cores[1, 1])
        with self.assertRaises(ValueError):
            find_ascent_cores(np.array([0.3, 0.4]))
        with self.assertRaises(ValueError):
            find_ascent_cores(ascent, neighborhood=2)
        with self.assertRaises(ValueError):
            find_ascent_cores(ascent, threshold=-0.1)


if __name__ == "__main__":
    unittest.main()
