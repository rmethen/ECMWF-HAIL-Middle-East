# ECMWF-HAIL-Middle-East
Experimental ECMWF severe-weather diagnostics for the Middle East.

Outputs:

- `ECMWF_HAIL_INDEX_MIDDLE_EAST_LATEST.png`: CAPE-free large-hail environment
  potential using
  hail-growth-layer depth, approximate wet-bulb-zero height, deep-layer wind
  difference, 700–500 hPa lapse rate, 500 hPa temperature, 850 hPa moisture,
  850 hPa lifted-index proxy and 700/500 hPa omega.
- `ECMWF_THUNDERSTORM_LIGHTNING_POTENTIAL_LATEST.png`: CAPE-free thunderstorm
  and lightning-potential proxy using LI, 700/500 hPa omega, moisture, lapse
  rate and shear. Light 500-hPa height contours help identify upper shortwaves.

Both products use real ECMWF Open Data fields over forecast hours 0–72. They
are experimental diagnostics and are not official ECMWF products. The
thunderstorm product is a potential index, not direct lightning-flash density.
The hail index highlights environments supportive of large hail; it is not a
forecast of hailstone diameter in centimetres.

## Hail research diagnostics (not used in the public map)

A manually dispatched hail-map run archives `hail_research_experiment.npz` for 30 days. It
contains the unchanged V2 score, `wmaxshear_proxy = sqrt(2*MUCAPE) ×
|V300−V850|` (m²/s²), and a unitless HGL × WBZ survival/growth interaction.
The proxy uses ECMWF most-unstable CAPE and 850–300 hPa bulk wind difference:
it is **not** the mixed-layer, effective-shear WMAXSHEAR in the Spanish study.
Neither candidate is calibrated or used in the operational V2 map. Archive
artifacts externally for a longer verification period if needed.

For documented hail events, prepare a CSV with
`event_id,valid_utc,latitude,longitude,hail_mm` (UTC ISO timestamps and
independently observed maximum diameter in mm). Use confirmed non-hail
thunderstorms with `hail_mm=0` as controls. Then run:

```sh
python verify_hail_events.py observations.csv data/hail_research_experiment.npz \
  --output data/hail_event_comparison.csv
```

The output gives values at the nearest grid cell and the maximum within
75 km and ±3 hours, plus the offset of that local maximum. Compare matched
hail and storm-control distributions, missed events and displaced forecasts
before considering any weight changes. A grid-cell maximum is not the
predicted hail-swath location, and the current ECMWF Open Data archive cannot
reconstruct the October 2019 Al-Wafra case from the live feed.
