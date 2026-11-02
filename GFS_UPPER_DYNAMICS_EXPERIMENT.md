# Experimental GFS upper-air diagnostics — Kuwait and Gulf

Source reference: [Alicia Bentley, University at Albany real-time GFS maps](https://www.atmos.albany.edu/student/abentley/realtime/standard.php?domain=india&variable=irro_wind). We **do not** scrape the site's map images. Meteorological fields are downloaded independently from [NOAA NOMADS GFS 0.25-degree GRIB2 filter](https://nomads.ncep.noaa.gov/gribfilter.php?ds=gfs_0p25).

## First-stage implementation

File: `gfs_upper_dynamics.py`. It produces three-panel experimental forecast images for Kuwait and the Gulf:

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

Real NOAA test (forecast hours +3, +6 and +12 from **one** cycle; download pressure-level fields and produce three PNGs; only run when external access is available):

```bash
python -m pip install numpy scipy xarray cfgrib eccodes matplotlib cartopy
python gfs_upper_dynamics.py --hours 3,6,12
```

Specify an initialization cycle in UTC if needed:

```bash
python gfs_upper_dynamics.py --cycle 2026110100 --hour 6
```

For an already-downloaded GFS subset:

```bash
python gfs_upper_dynamics.py --cycle 2026110100 --hour 6 --source data/sample.grib2
```

Expected multi-lead outputs: `output/GFS_UPPER_DYNAMICS_KUWAIT_GULF_EXPERIMENTAL_F003.png`, `_F006.png` and `_F012.png`. (The `_F` suffix is added to the full base name.) The latest-cycle fallback checks up to six 6-hourly cycles, and only accepts a cycle if **all** requested hours are available; otherwise the series fails clearly. A single-lead `--hour 6` remains supported and produces the original single-file name. The downloaded GRIB files and PNGs are not automatically committed or published on the project site. The workflow uploads the PNGs as a three-day GitHub Actions artifact, not a live public map.

## Map presentation and geographic fidelity

- Natural Earth 110m **real coastlines and international borders** are rendered via Cartopy; first use may need network access to download public-domain geographic shapes.
- Kuwait, Basra, Doha, Dubai, Riyadh, and Muscat are labeled for geographic context; city point locations are reference markers, not storm observations.
- Display-only Gaussian smoothing of approximately **0.75° standard deviation** reduces pixel-scale derivative noise. Calculated diagnostics and physical units remain unchanged. Small, localized maxima can be attenuated, so the display is suitable for synoptic/mid-scale context rather than exact cell-scale storm analysis.
- The vorticity panel uses a sequential 0–30 × 10⁻⁵ s⁻¹ color scale: stronger absolute vorticity does not itself prove severe convection. Divergence and omega remain zero-centered. A positive *upward omega* value means negative original pressure vertical velocity (physical ascent).
- Three lead-time maps are same-cycle and show forecast evolution, **not radar or 1–3-hour observed nowcasting**.

## Safety gate and next steps

The experimental workflow runs offline tests on the feature branch and attempts a NOAA same-cycle +3/+6/+12 h smoke test, uploading only a short-lived GitHub Actions image artifact. It has **read-only contents permissions**, never commits generated data, never modifies `main`, and never overwrites existing ECMWF hail, SHIP, Kelvin, or flood products.

Before operational adoption: verify real GRIB inventory, confirm units and sign conventions against independent charts, inspect behavior in several observed storm/non-storm cases, and review geographic contours. Only after that consider Helmholtz-decomposed irrotational wind and Ertel PV with appropriate temperature/pressure-level data, boundary assumptions, and validation.
