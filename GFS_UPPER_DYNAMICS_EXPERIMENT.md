# Experimental GFS upper-air diagnostics — Kuwait and Gulf

Source reference: [Alicia Bentley, University at Albany real-time GFS maps](https://www.atmos.albany.edu/student/abentley/realtime/standard.php?domain=india&variable=irro_wind). We **do not** scrape the site's map images. Meteorological fields are downloaded independently from [NOAA NOMADS GFS 0.25-degree GRIB2 filter](https://nomads.ncep.noaa.gov/gribfilter.php?ds=gfs_0p25).

## First-stage implementation

File: `gfs_upper_dynamics.py`. It produces three-panel experimental forecast images for Kuwait and the Gulf:

- 250-hPa **full-wind horizontal divergence**, derived from GFS U and V on a spherical latitude/longitude grid (10⁻⁵ s⁻¹).
- Mean 600/500/400-hPa **upward omega**, displayed positive for ascent (Pa/s). This is the negative of the equally weighted mean of three pressure-level omega fields, not an integrated vertical velocity.
- 500-hPa **relative vorticity** (10⁻⁵ s⁻¹), derived from 500-hPa GFS winds; this corresponds to the type of field used in the Bentley 500-hPa Relative Vorticity panel. The scientific calculation also retains **absolute vorticity** including the Coriolis parameter for future diagnostics, but does not plot it.

The grid is approximately 0.25 degrees, *not* 0.25 km. The fields are forecasts, not radar observations or verified hail detections. They may help diagnose forcing but cannot alone establish CIN erosion or convective initiation.

**Important scientific boundary:** Upper-level full-wind divergence is **not** the irrotational-wind vector product displayed by Bentley. Neither full-wind divergence nor 500-hPa relative/absolute vorticity is equivalent to an Ertel-PV 300–200-hPa product. Genuine PV and irrotational wind require additional ingredients and validation and have deliberately **not** been labeled as implemented here.

## Run and verification

Offline numerical tests (no NOAA request):

```bash
python -m pip install numpy scipy
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
- **Experimental red dots** on the 600–400 hPa omega panel identify *local maxima of unsmoothed* mean upward omega (positive = ascent), subject to a **0.50 Pa/s** threshold and a **9×9 GFS-gridpoint** maximum filter (roughly 2.25° across at nominal 0.25° spacing). Flat maxima collapse to one dot. Dot count is limited to the map extent. This threshold is a visualization starting point, not validated as a heavy-rain discriminator, and the locations can differ from maxima of the smoothed background shading.
- The Albany legend's red-ascent units `5 × 10⁻³ hPa/s` correspond to `0.5 Pa/s`; the new **0.50 Pa/s** dot threshold is comparable in nominal omega magnitude but **not** an exact replication of Albany's red-ascent product (different extraction and spatial processing). The wider local-maximum window suppresses duplicate nearby dots. Neither the dots nor the model omega field confirms heavy precipitation without moisture, instability, dynamics and observation checks.
- The **relative-vorticity** panel and the divergence panel use diverging zero-centered −10 to +10 × 10⁻⁵ s⁻¹ scales. Positive relative vorticity (in the Northern Hemisphere) can identify cyclonic shortwaves but does not itself prove severe convection. The omega panel is also zero-centered; a positive *upward omega* value means negative original pressure vertical velocity (physical ascent).
- Three lead-time maps are same-cycle and show forecast evolution, **not radar or 1–3-hour observed nowcasting**.

## Mobile-view color and 250-hPa wind arrows

The enlarged Gulf-only map now uses display normalization **−0.40 to +0.40 Pa/s** so ascent (red) and descent (blue) are easier to see on a phone. Values beyond the endpoints saturate in color **without changing the underlying GFS fields, core positions or 0.50 Pa/s threshold**; numeric labels remain unsaturated.

The arrow overlay is a sparse ~2.5-degree sample of the authentic NOAA GFS **250-hPa full horizontal wind vectors** (`u250_ms`, `v250_ms`) already contained in the downloaded GRIB. The direction and speed are measured in m/s; the key displays 20 m/s. These are **not** irrotational (divergent-component) vectors, and they should not be interpreted as direct proof of upper-level divergence or vertical ascent. The 250-hPa full-wind divergence calculated separately remains on the original three-panel diagnostic graphic. The mobile view always distinguishes this limitation in the label.

## Experimental bounded-domain irrotational winds at 250 hPa

We now prototype the **regional Helmholtz gradient (irrotational/potential) component** using only real NOAA GFS pressure-level winds at 250 hPa. This is NOT a global spectral Helmholtz decomposition and does NOT exactly reproduce the University at Albany irrotational-wind product.

- In the full domain **30–68°E, 10–44°N**, calculate spherical horizontal divergence of the GFS 250-hPa U/V field.
- Solve the spherical Poisson equation `Laplacian(chi) = divergence` on the enlarged model grid with **zero velocity-potential at the rectangular boundary** using SciPy sparse linear algebra. The resulting `gradient(chi)` is irrotational to numerical precision away from grid edges.
- Draw only the interior Gulf view **42–59°E, 22–36°N**. A broad buffer is intentional because finite-domain boundary assumptions can strongly affect the arrows.
- Use a **10 m/s vector key** for regional potential winds, which differ substantially from the full-wind direction and magnitude. Red/blue omega and the red/yellow cores are not modified.
- The solution is non-unique without specified boundary conditions. The imposed Dirichlet potential boundary can suppress real large-scale harmonic/divergent components, so **do not treat these arrows as globally valid upper-air divergence vectors or operational evidence of convection**.
- Offline numerical gates include uniform non-divergent flow, recovery of a known synthetic velocity potential, reversed coordinate order, and invalid input rejection. After passing them, examine real-data Gulf maps and compare against published large-scale divergence patterns. Only use an operational/global method after global winds and a validated spectral Helmholtz solver are available.

## Safety gate and next steps

The experimental workflow runs offline tests on the feature branch and attempts a NOAA same-cycle +3/+6/+12 h smoke test, uploading only a short-lived GitHub Actions image artifact. It has **read-only contents permissions**, never commits generated data, never modifies `main`, and never overwrites existing ECMWF hail, SHIP, Kelvin, or flood products.

Before operational adoption: verify real GRIB inventory, confirm units and sign conventions against independent charts, inspect behavior in several observed storm/non-storm cases, and review geographic contours. Only after that consider Helmholtz-decomposed irrotational wind and Ertel PV with appropriate temperature/pressure-level data, boundary assumptions, and validation.
