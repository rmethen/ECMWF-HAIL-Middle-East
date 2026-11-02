# Experimental GFS upper-air diagnostics — Kuwait and Gulf

Source reference: [Alicia Bentley, University at Albany real-time GFS maps](https://www.atmos.albany.edu/student/abentley/realtime/standard.php?domain=india&variable=irro_wind). We **do not** scrape the site's map images. Meteorological fields are downloaded independently from [NOAA NOMADS GFS 0.25-degree GRIB2 filter](https://nomads.ncep.noaa.gov/gribfilter.php?ds=gfs_0p25).

## First-stage implementation

File: `gfs_upper_dynamics.py`. It produces one three-panel experimental forecast image for Kuwait and the Gulf:

- 250-hPa **full-wind horizontal divergence**, derived from GFS U and V on a spherical latitude/longitude grid (10⁻⁵ s⁻¹).
- Mean 600/500/400-hPa **upward omega**, displayed positive for ascent (Pa/s). This is the negative of the equally weighted mean of three pressure-level omega fields, not an integrated vertical velocity.
- 500-hPa **absolute vorticity**, derived from 500-hPa winds including the Coriolis parameter (10⁻⁵ s⁻¹).

The grid is approximately 0.25 degrees, *not* 0.25 km. The fields are forecasts, not radar observations or verified hail detections. They may help diagnose forcing but cannot alone establish CIN erosion or convective initiation.

**Important scientific boundary:** Upper-level full-wind divergence is **not** the irrotational-wind vector product displayed by Bentley. Neither full-wind divergence nor 500-hPa absolute vorticity is equivalent to an Ertel-PV 300–200-hPa product. Genuine PV and irrotational wind require additional ingredients and validation and have deliberately **not** been labeled as implemented here.

## Run and verification

Offline numerical tests (no NOAA request):

```bash
python -m pip install numpy
python -m unittest discover -s tests -p 'test_gfs_upper_dynamics.py' -v
```

Real one-time NOAA test (forecast hour +6; downloads several pressure-level fields and produces one PNG; only run when external access is available):

```bash
python -m pip install numpy xarray cfgrib eccodes matplotlib
python gfs_upper_dynamics.py --hour 6
```

Specify an initialization cycle in UTC if needed:

```bash
python gfs_upper_dynamics.py --cycle 2026110100 --hour 6
```

For an already-downloaded GFS subset:

```bash
python gfs_upper_dynamics.py --cycle 2026110100 --hour 6 --source data/sample.grib2
```

Expected output: `output/GFS_UPPER_DYNAMICS_KUWAIT_GULF_EXPERIMENTAL.png`. The latest-cycle fallback checks up to six 6-hourly cycles and fails clearly if NOAA is unavailable. The downloaded GRIB file and PNG are not automatically committed or published on the project site.

## Safety gate and next steps

The experimental workflow runs offline tests on the feature branch and attempts a one-time NOAA +6 h smoke test, uploading only a short-lived GitHub Actions image artifact. It has **read-only contents permissions**, never commits generated data, never modifies `main`, and never overwrites existing ECMWF hail, SHIP, Kelvin, or flood products.

Before operational adoption: verify real GRIB inventory, confirm units and sign conventions against independent charts, inspect behavior in several observed storm/non-storm cases, and review geographic contours. Only after that consider Helmholtz-decomposed irrotational wind and Ertel PV with appropriate temperature/pressure-level data, boundary assumptions, and validation.
