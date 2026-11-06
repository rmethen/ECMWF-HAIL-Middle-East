"""End-to-end GFS SHIP computation from model profiles (experimental).

Uses MetPy's most_unstable_cape_cin on pressure-level soundings. Strictly
masks columns lacking near-surface pressure-level coverage. Never substitutes
a fixed-layer shear proxy for 0–6 km AGL shear.
"""
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
from metpy.units import units
import metpy.calc as mpcalc
from gfs_ship_map import ShipInputs, compute_ship
from render_gfs_ship_index import render_ship

def field(path, short_name, level_type):
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={
        "filter_by_keys": {"shortName": short_name, "typeOfLevel": level_type},
        "indexpath": ""}) as ds:
        return ds[short_name].load()

def interpolate_height(z, values, target):
    good = np.isfinite(z) & np.isfinite(values)
    if good.sum() < 2 or target < np.min(z[good]) or target > np.max(z[good]):
        return np.nan
    return np.interp(target, z[good], values[good])

def column(p, t, q, z, u, v, surface_p, terrain):
    """Return physically derived six inputs, or None if profile incomplete."""
    ok = np.isfinite(p+t+q+z+u+v) & (p <= surface_p / 100)
    if ok.sum() < 8:
        return None
    p, t, q, z, u, v = [a[ok] for a in (p,t,q,z,u,v)]
    order = np.argsort(p)[::-1]
    p,t,q,z,u,v = [a[order] for a in (p,t,q,z,u,v)]
    # No trustworthy lowest-300-hPa parcel search if first level too high AGL.
    if z[0] - terrain > 650 or z[-1] - terrain < 6000:
        return None
    if not np.all(np.diff(p) < 0):
        return None
    q = np.clip(q, 1e-8, .04)
    dew = mpcalc.dewpoint_from_specific_humidity(p*units.hPa, t*units.kelvin,
                                                 q*units.dimensionless)
    try:
        cape, _ = mpcalc.most_unstable_cape_cin(
            p*units.hPa, t*units.kelvin, dew, depth=300*units.hPa)
        mu_idx = int(np.argmax(mpcalc.equivalent_potential_temperature(
            p[:min(len(p), 9)]*units.hPa,
            t[:min(len(p), 9)]*units.kelvin, dew[:min(len(p), 9)]).magnitude))
        mu_mix = q[mu_idx] / (1-q[mu_idx]) * 1000
        t500 = float(np.interp(500, p[::-1], t[::-1]) - 273.15)
        t700 = float(np.interp(700, p[::-1], t[::-1]))
        z500 = float(np.interp(500, p[::-1], z[::-1]))
        z700 = float(np.interp(700, p[::-1], z[::-1]))
        lapse = (t700 - (t500+273.15)) / ((z500-z700)/1000)
        u0 = interpolate_height(z,u,terrain)
        v0 = interpolate_height(z,v,terrain)
        # If lowest model level above ground, use lowest model wind only if <=150m AGL.
        if not np.isfinite(u0) and z[0]-terrain <= 150:
            u0,v0 = u[0],v[0]
        u6 = interpolate_height(z,u,terrain+6000)
        v6 = interpolate_height(z,v,terrain+6000)
        shear = float(np.hypot(u6-u0,v6-v0))
        freezing = np.nan
        for k in range(len(z)-1):
            if t[k] >= 273.15 and t[k+1] < 273.15:
                freezing = max(0.,float(np.interp(273.15,[t[k+1],t[k]],[z[k+1],z[k]])-terrain))
                break
        vals = (max(0.,float(cape.magnitude)),mu_mix,lapse,t500,shear,freezing)
        return vals if np.all(np.isfinite(vals)) else None
    except (ValueError, IndexError, ZeroDivisionError):
        return None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--profile",default="data/gfs_ship_f006.grib2")
    parser.add_argument("--surface",default="data/gfs_ship_surface_f006.grib2")
    args=parser.parse_args()
    profile=args.profile
    t=field(profile,"t","isobaricInhPa")
    q=field(profile,"q","isobaricInhPa")
    z=field(profile,"gh","isobaricInhPa")
    u=field(profile,"u","isobaricInhPa")
    v=field(profile,"v","isobaricInhPa")
    sp=field(args.surface,"sp","surface").values
    terrain=field(args.surface,"orog","surface").values
    p=t.isobaricInhPa.values.astype(float)
    lon=t.longitude.values;lat=t.latitude.values
    shape=t.shape[1:]
    out=np.full(shape,np.nan)
    valid=0
    for i in range(shape[0]):
        for j in range(shape[1]):
            inputs=column(p,t.values[:,i,j],q.values[:,i,j],z.values[:,i,j],
                          u.values[:,i,j],v.values[:,i,j],sp[i,j],terrain[i,j])
            if inputs is None: continue
            try:
                arrays=[np.array([x],dtype=float) for x in inputs]
                out[i,j]=compute_ship(ShipInputs(*arrays))[0]
                valid+=1
            except ValueError:
                continue
    if valid < 100:
        raise RuntimeError(f"Only {valid} valid SHIP cells; refusing misleading map")
    valid_time=np.datetime_as_string(t.valid_time.values,unit="h")
    cycle=np.datetime_as_string(t.time.values,unit="h")
    dest=render_ship(lon,lat,out,cycle_utc=cycle,valid_utc=valid_time)
    print(f"SHIP map {dest}; valid columns {valid}/{out.size}")

if __name__=="__main__":
    main()
