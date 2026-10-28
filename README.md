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

The primary hail map also shows a dotted GFS forecast storm-support overlay
when a GFS run matches the ECMWF peak valid time (simulated reflectivity
≥35 dBZ and instantaneous rain rate ≥1 mm/h). It is independent model
guidance, not observed radar or hail confirmation; the ECMWF hail index
and its score are unchanged if GFS is unavailable.

`gfs_hail_map.py` also builds a **separate** American model hail-environment
map from GFS pressure-level temperature, height, humidity, wind, and vertical
velocity. It applies the same experimental V2 ingredients and fixed thresholds
as the ECMWF hail map across +6 to +72 hours, then publishes
`output/GFS_HAIL_INDEX_MIDDLE_EAST_LATEST.png`. The two models are not
observations and their index values are not calibrated hail probabilities.
