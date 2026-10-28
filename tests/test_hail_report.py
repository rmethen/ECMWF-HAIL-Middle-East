"""Check city/time alignment, missing cells and forecast-step interpretation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from hail_report import (
    DIAGNOSTICS, city_diagnostics, forecast_hours, format_time, peak_fields,
    valid_time, write_city_csv,
)


class HailReportTests(unittest.TestCase):
    def test_local_peaks_differ_from_domain_peak_and_keep_missing_cells(self):
        values = np.array([[[90, 10, np.nan]], [[30, 50, np.nan]],
                           [[90, 70, np.nan]]])
        _, maxima, peak = peak_fields(values)
        np.testing.assert_allclose(maxima, [[90, 70, np.nan]], equal_nan=True)
        np.testing.assert_array_equal(peak, [[0, 2, 0]])
        with self.assertRaisesRegex(ValueError, "No finite"):
            peak_fields(np.full((3, 2, 2), np.nan))

    def test_numeric_and_timedelta_steps_match_and_reject_mismatch(self):
        for steps in (np.array([0, 6, 18]), np.array([0, 6, 18], dtype="timedelta64[h]")):
            np.testing.assert_equal(forecast_hours(steps, 3), [0, 6, 18])
        for steps in ([0, 6], [0, 6, 6], [6, 0, 18], [0, np.nan, 18]):
            with self.assertRaises(ValueError):
                forecast_hours(steps, 3)

    def test_local_time_rolls_over_date(self):
        time = valid_time(np.array("2026-10-28T12:00", dtype="datetime64[m]"), 12)
        self.assertEqual(format_time(time), "2026-10-29 00:00 UTC")
        self.assertEqual(format_time(time, local=True), "2026-10-29 03:00 UTC+3")
        self.assertEqual(format_time(np.datetime64("NaT")), "N/A")

    def test_city_ingredients_follow_own_peak_not_each_ingredient_maximum(self):
        # Descending latitude, as in ECMWF; cities map to different cells.
        data = dict(latitude=np.array([32., 31., 30., 29.]),
                    longitude=np.array([37., 39., 41., 48.]),
                    init_time=np.datetime64("2026-10-28T00:00"))
        index = np.full((3, 4, 4), 10.)
        index[1, 1, 2] = 60.  # Arar: +6, not domain peak.
        index[2, 0, 1] = 70.  # Turaif: +18.
        index[0, 3, 3] = 95.  # Domain maximum at Kuwait: +0.
        for key, _, _ in DIAGNOSTICS:
            data[key] = np.broadcast_to(np.array([900., 100., 200.])[:, None, None],
                                        index.shape).copy()
        counts = np.broadcast_to(np.array([5, 2, 4])[:, None, None], index.shape)
        rows = city_diagnostics(data, index, np.array([0., 6., 18.]), counts,
                                np.max(index, axis=0) - 1.)
        arar, turaif = rows[:2]
        self.assertEqual((arar["peak_hour"], turaif["peak_hour"]), (6, 18))
        self.assertEqual((arar["wbz_m"], turaif["wbz_m"]), (100, 200))
        self.assertEqual((arar["strong_ingredients"], turaif["strong_ingredients"]), (2, 4))
        self.assertEqual(arar["v2"], 60)
        self.assertEqual(arar["map_v2"], 59)
        self.assertEqual(arar["peak_local"], "2026-10-28 09:00 UTC+3")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cities.csv"
            write_city_csv(path, rows, data["init_time"], [0, 6, 18], "original")
            text = path.read_text()
            self.assertIn("period_end_utc", text)
            self.assertIn("Arar", text)

        index[:, 1, 2] = np.nan
        missing = city_diagnostics(data, index, np.array([0., 6., 18.]), counts,
                                   np.zeros((4, 4)))[0]
        self.assertEqual(missing["status"], "no finite V2")
        data["longitude"] = np.array([20., 21., 22., 23.])
        outside = city_diagnostics(data, index, np.array([0., 6., 18.]), counts,
                                   np.zeros((4, 4)))
        self.assertTrue(all(row["status"] == "outside grid" for row in outside))

    def test_single_time_2d_input(self):
        index, maximum, peak = peak_fields(np.array([[21., 30.]]))
        self.assertEqual(index.shape, (1, 1, 2))
        np.testing.assert_equal(maximum, [[21, 30]])
        np.testing.assert_equal(peak, [[0, 0]])

    def test_both_variants_render_reports_without_real_weather_download(self):
        from unittest.mock import patch
        import matplotlib
        matplotlib.use("Agg")
        from cartopy.mpl.geoaxes import GeoAxes
        import make_hail_map as renderer

        shape = (3, 4, 4)
        data = dict(latitude=np.linspace(33, 28, 4), longitude=np.linspace(36, 49, 4),
                    steps=np.array([0, 6, 18], dtype="timedelta64[h]"),
                    init_time=np.datetime64("2026-10-28T00:00"))
        inputs = dict(shear_850_300=23., wbz_m=2600., hgl_depth_m=4400.,
                      lapse_700_500=7., t500_c=-20., moisture_850=11.,
                      li850=-3., omega_700=-0.8, omega_500=-0.6)
        data.update({key: np.full(shape, value) for key, value in inputs.items()})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "synthetic.npz"
            np.savez(fixture, **data)
            # Skip only external geography datasets; exercise real rendering,
            # contours, time conversion, city table and all output writers.
            with patch.object(renderer, "DATA_FILE", fixture), \
                    patch.object(GeoAxes, "add_feature", return_value=None):
                for variant in ("original", "1500-3000"):
                    renderer.main(wbz_variant=variant, output_file=root / f"{variant}_LATEST.png",
                                  show_extreme=variant != "original",
                                  show_omega=variant != "original")
                    for suffix in ("LATEST.png", "PEAK_TIME_LATEST.png",
                                   "CITY_DIAGNOSTICS_LATEST.png", "CITY_DIAGNOSTICS_LATEST.csv"):
                        self.assertGreater((root / f"{variant}_{suffix}").stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
