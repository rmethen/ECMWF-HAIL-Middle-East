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
