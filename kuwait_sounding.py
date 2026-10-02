"""Single-point experimental Kuwait forecast sounding from ECMWF Open Data."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from metpy.plots import SkewT
import metpy.calc as mpcalc
from metpy.units import units

from ecmwf_hail_data import LEVELS, PRESSURE_PARAMS

LAT, LON = 29.38, 47.98


def fetch(step, directory):
    from ecmwf.opendata import Client
    client = Client(source='ecmwf', model='ifs', resol='0p25')
    directory.mkdir(parents=True, exist_ok=True)
    upper = directory / 'kuwait_pressure.grib2'
    surface = directory / 'kuwait_surface.grib2'
    # Open Data serves global fields; nearest Kuwait grid point is extracted locally.
    common = dict(time=0, type='fc', stream='oper', step=step)
    client.retrieve(**common, levelist=LEVELS,
                    param=[p for p in PRESSURE_PARAMS if p in ('t', 'u', 'v', 'q', 'gh')],
                    target=str(upper))
    client.retrieve(**common, param=['2t', '2d', '10u', '10v', 'msl'], target=str(surface))
    return upper, surface


def field(path, name, level_type=None):
    filt = {'shortName': name}
    if level_type:
        filt['typeOfLevel'] = level_type
    ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={
        'filter_by_keys': filt, 'indexpath': ''})
    return ds[list(ds.data_vars)[0]]


def point(da, step):
    if 'step' in da.dims:
        da = da.sel(step=np.timedelta64(step, 'h'))
    return da.sel(latitude=LAT, longitude=LON, method='nearest')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=24)
    parser.add_argument('--pressure-file', type=Path)
    parser.add_argument('--surface-file', type=Path)
    parser.add_argument('--output', type=Path, default=Path('output/KUWAIT_SKEWT_LATEST.png'))
    args = parser.parse_args()
    if (args.pressure_file is None) != (args.surface_file is None):
        parser.error('Supply both GRIB files or neither')
    if args.pressure_file:
        upper, surface = args.pressure_file, args.surface_file
    else:
        upper, surface = fetch(args.step, Path('data'))
    pl = {n: point(field(upper, n, 'isobaricInhPa'), args.step)
          for n in ('t', 'q', 'u', 'v', 'gh')}
    sf = {n: point(field(surface, n), args.step)
          for n in ('2t', '2d', '10u', '10v', 'msl')}
    t = pl['t']
    initial = np.datetime_as_string(t.time.values, unit='m')
    valid = np.datetime_as_string(t.valid_time.values, unit='m')
    lat, lon = float(t.latitude), float(t.longitude)
    levels = np.asarray(t.isobaricInhPa, dtype=float)
    order = np.argsort(levels)[::-1]
    p = levels[order] * units.hPa
    temp = np.asarray(t)[order] * units.kelvin
    q = np.asarray(pl['q'])[order] * units('kg/kg')
    dew = mpcalc.dewpoint_from_specific_humidity(p, temp, q)
    u = np.asarray(pl['u'])[order] * units('m/s')
    v = np.asarray(pl['v'])[order] * units('m/s')
    z = np.asarray(pl['gh'])[order] * units.meter
    # MSLP approximates station pressure at this low-elevation coastal point.
    ps = float(sf['msl']) / 100 * units.hPa
    if ps.magnitude > p[0].magnitude:
        p = np.concatenate(([ps.magnitude], p.magnitude)) * units.hPa
        temp = np.concatenate(([float(sf['2t'])], temp.magnitude)) * units.kelvin
        dew = np.concatenate(([float(sf['2d'])], dew.to('kelvin').magnitude)) * units.kelvin
        u = np.concatenate(([float(sf['10u'])], u.magnitude)) * units('m/s')
        v = np.concatenate(([float(sf['10v'])], v.magnitude)) * units('m/s')
        z = np.concatenate(([0.], z.magnitude)) * units.meter
    else:
        print('MSLP <= 1000 hPa: using 1000 hPa as lowest parcel level')
    parcel = mpcalc.parcel_profile(p, temp[0], dew[0]).to('degC')
    cape, cin = mpcalc.cape_cin(p, temp, dew, parcel)
    lcl_p, _ = mpcalc.lcl(p[0], temp[0], dew[0])
    lfc_p, _ = mpcalc.lfc(p, temp, dew, parcel_temperature_profile=parcel)
    pw = mpcalc.precipitable_water(p, dew)
    shear = mpcalc.bulk_shear(p, u, v, height=z, depth=6000 * units.meter)
    shear_mag = np.hypot(shear[0], shear[1]).to('m/s')
    wind10 = np.hypot(float(sf['10u']), float(sf['10v']))
    def fmt(value, spec='.0f'):
        return format(value.magnitude, spec) if np.isfinite(value.magnitude) else 'N/A'
    fig = plt.figure(figsize=(10, 9), facecolor='white')
    skew = SkewT(fig, rotation=45, rect=(.08, .12, .66, .74))
    skew.plot(p, temp.to('degC'), 'r', linewidth=2, label='Temperature')
    skew.plot(p, dew.to('degC'), 'g', linewidth=2, label='Dew point (q-derived aloft)')
    skew.plot(p, parcel, color='black', linewidth=1.3, label='Surface parcel')
    skew.plot_barbs(p[::2], u[::2].to('knots'), v[::2].to('knots'))
    skew.ax.set_ylim(1050, 200)
    skew.ax.set_xlim(-45, 50)
    skew.ax.set_xlabel('Temperature (°C)')
    skew.ax.set_ylabel('Pressure (hPa)')
    skew.ax.legend(loc='upper right', fontsize=8)
    skew.plot_dry_adiabats(alpha=.25)
    skew.plot_moist_adiabats(alpha=.25)
    skew.plot_mixing_lines(alpha=.25)
    fig.suptitle(f'ECMWF experimental Kuwait Skew-T | {lat:.2f}°N {lon:.2f}°E', fontsize=13)
    fig.text(.09, .91, f'Init: {initial} UTC     Valid: {valid} UTC     Lead: +{args.step} h', fontsize=10)
    summary = (f'CAPE  {fmt(cape)} J/kg\nCIN  {fmt(cin)} J/kg\n'
               f'LCL  {fmt(lcl_p)} hPa\nLFC  {fmt(lfc_p)} hPa\n'
               f'PW  {fmt(pw.to("mm"), ".1f")} mm\n'
               f'10 m wind  {wind10:.1f} m/s\n0–6 km shear  {fmt(shear_mag, ".1f")} m/s')
    fig.text(.75, .67, summary, fontsize=10, va='top', linespacing=1.7)
    fig.text(.08, .035, 'Forecast model profile • surface pressure approximated by MSLP • sparse vertical levels; diagnostics are indicative.', fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    metadata = {
        'init_utc': initial + 'Z',
        'valid_utc': valid + 'Z',
        'generated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'forecast_hour': args.step,
    }
    metadata_path = args.output.with_suffix('.json')
    temporary_path = metadata_path.with_suffix('.json.tmp')
    temporary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary_path.replace(metadata_path)
    print(f'{args.output}\n{summary}\nInit {initial} UTC / Valid {valid} UTC / grid {lat:.2f}, {lon:.2f}')


if __name__ == '__main__':
    main()
