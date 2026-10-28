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

## Hail time and city diagnostics

Both V2 variants retain the same index calculation. Their main maps show the
smoothed maximum over all available forecast times, with the actual forecast
period stated explicitly. The domain-wide raw maximum time is not the peak
time of every city. All local dates use UTC+3 (Saudi Arabia and Kuwait).

Each run also generates, for each variant:

- `*_PEAK_TIME_LATEST.png`: first sampled local maximum time, in hours from
  initialization. Cells with raw maximum V2 below 20 or no data are blank.
- `*_CITY_DIAGNOSTICS_LATEST.png`: ingredients at Arar and Turaif's own raw V2
  peak times, not at the domain-wide peak or at each ingredient's separate peak.
- `*_CITY_DIAGNOSTICS_LATEST.csv`: nearest-grid diagnostics for Arar, Turaif,
  Qurayyat, Sakaka and Kuwait, including sampled grid coordinates, raw V2,
  smoothed map V2, lead time, UTC/local date, and each raw ingredient.

Tables use raw nearest-grid values; smoothed map colors can differ. Ties use
the first sampled time. The data processor computes approximate WBZ using
geopotential heights, so its units are metres above mean sea level, not ground.
`moisture_850` stores specific humidity in g/kg, and `shear_850_300` is the
850-to-300-hPa wind difference, not a measured surface-to-6-km shear.
The strong-ingredient count remains experimental and uncalibrated.

Run the reporting checks with:
`python -m unittest discover -s tests -p 'test_hail_report.py'`.

