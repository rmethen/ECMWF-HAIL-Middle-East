"""Experimental GFS upper-air dynamics for Kuwait and the Gulf.

Independent of operational ECMWF and GFS hail indices.
Computes 250-hPa horizontal divergence, 600–400-hPa mean upward motion,
and 500-hPa absolute vorticity from *real* NOAA GFS pressure-level fields.

This is NOT the Bentley irrotational-wind or Ertel-PV product, and NOT radar.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

NOMADS = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
EARTH_RADIUS_M = 6_371_000.0
EARTH_ROTATION_RAD_S = 7.2921159e-5
# Larger download region keeps finite-difference boundaries away from the map.
FETCH_BOUNDS = (30, 68, 10, 44)  # west, east, south, north (degrees)
PLOT_BOUNDS = (35, 64, 16, 38)
GRIB_PATH = Path("data/gfs_upper_dynamics_f006.grib2")
PLOT_PATH = Path("output/GFS_UPPER_DYNAMICS_KUWAIT_GULF_EXPERIMENTAL.png")


def build_url(cycle: datetime, hour: int = 6) -> str:
    """Build a bounded 0.25-degree NOMADS GRIB2 query; no image scraping."""
    if cycle.tzinfo is None or cycle.utcoffset() != timedelta(0):
        raise ValueError("cycle must be timezone-aware UTC")
    if cycle.hour not in (0, 6, 12, 18) or cycle.minute or cycle.second:
        raise ValueError("GFS initialization cycle must be 00/06/12/18 UTC")
    if hour < 0 or hour > 120 or not isinstance(hour, int):
        raise ValueError("forecast hour must be an integer from 0 to 120")
    west, east, south, north = FETCH_BOUNDS
    query = {
        "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{hour:03d}",
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
        "subregion": "",
        "leftlon": west, "rightlon": east, "bottomlat": south,
        "toplat": north,
    }
    for variable in ("UGRD", "VGRD", "VVEL"):
        query[f"var_{variable}"] = "on"
    for level in (250, 400, 500, 600):
        query[f"lev_{level}_mb"] = "on"
    return NOMADS + "?" + urlencode(query)


def download(cycle: datetime, hour: int = 6, path: Path = GRIB_PATH) -> Path:
    """Stream one authentic GFS GRIB2 subset; reject HTML/error responses."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    try:
        req = Request(build_url(cycle, hour),
                      headers={"User-Agent": "ECMWF-HAIL-Middle-East-upper-dynamics/1.0"})
        with urlopen(req, timeout=150) as response, part.open("wb") as handle:
            if response.read(4) != b"GRIB":
                raise ValueError("NOMADS did not return a GRIB2 message")
            handle.write(b"GRIB")
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if part.stat().st_size < 10_000:
            raise ValueError("Downloaded GFS subset is unexpectedly small")
        # A GRIB message must end with the '7777' terminator.
        with part.open("rb") as handle:
            handle.seek(-4, 2)
            if handle.read(4) != b"7777":
                raise ValueError("GRIB2 download is incomplete")
        part.replace(path)
        return path
    finally:
        part.unlink(missing_ok=True)


def _vector_derivatives(lon, lat, u, v):
    """Horizontal spherical divergence and relative vorticity (s^-1).

    Accepts monotonic ascending OR descending coordinates, lat not near poles.
    """
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    u, v = np.asarray(u, float), np.asarray(v, float)
    if u.shape != (len(lat), len(lon)) or v.shape != u.shape:
        raise ValueError("wind shapes must be (latitude, longitude)")
    if len(lon) < 3 or len(lat) < 3 or not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError("Need at least 3 valid coordinates along each axis")
    if np.any(np.diff(lon) == 0) or np.any(np.diff(lat) == 0):
        raise ValueError("No duplicate grid coordinates")
    if abs(lat).max() >= 85:
        raise ValueError("Spherical derivatives near poles are not supported")
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    cosphi = np.cos(lat_rad)[:, None]
    dudlon = np.gradient(u, lon_rad, axis=1, edge_order=2)
    dvdlat_cos = np.gradient(v * cosphi, lat_rad, axis=0, edge_order=2)
    dvdlon = np.gradient(v, lon_rad, axis=1, edge_order=2)
    dudlat_cos = np.gradient(u * cosphi, lat_rad, axis=0, edge_order=2)
    div = (dudlon + dvdlat_cos) / (EARTH_RADIUS_M * cosphi)
    vort = (dvdlon - dudlat_cos) / (EARTH_RADIUS_M * cosphi)
    return div, vort


def calculate_diagnostics(lon, lat, u250, v250, u500, v500,
                          w400, w500, w600):
    """Return meteorological fields on a single GFS lat/lon grid.

    omega is Pa/s; negative omega is ascent. Mean is an equally weighted
    three-level approximation, not a vertical integral.
    """
    div250, _ = _vector_derivatives(lon, lat, u250, v250)
    _, rel_vort500 = _vector_derivatives(lon, lat, u500, v500)
    coriolis = 2 * EARTH_ROTATION_RAD_S * np.sin(np.deg2rad(lat))[:, None]
    w = [np.asarray(item, dtype=float) for item in (w400, w500, w600)]
    if any(item.shape != div250.shape for item in w):
        raise ValueError("omega fields must match the wind grid")
    return {
        "div250_1e5_s": div250 * 1e5,
        "upward_omega_pa_s": -(w[0] + w[1] + w[2]) / 3,
        "abs_vort500_1e5_s": (rel_vort500 + coriolis) * 1e5,
    }


def _read_grib_field(path: Path, name: str, levels: tuple[int, ...]):
    """Load one exact pressure-level field from a multi-message GRIB2 file."""
    import xarray as xr

    with xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {
                "shortName": name, "typeOfLevel": "isobaricInhPa"
            },
            "indexpath": "",
        },
    ) as ds:
        if not ds.data_vars:
            raise ValueError(f"Missing GFS field {name}")
        da = ds[list(ds.data_vars)[0]]
        present = set(np.atleast_1d(da.isobaricInhPa.values).astype(int))
        if not set(levels).issubset(present):
            raise ValueError(f"Missing GFS pressure levels for {name}: {set(levels) - present}")
        return da.sel(isobaricInhPa=list(levels)).load()


def read_diagnostics(path: Path):
    u = _read_grib_field(path, "u", (250, 500))
    v = _read_grib_field(path, "v", (250, 500))
    w = _read_grib_field(path, "w", (400, 500, 600))
    for other in (v, w):
        if not np.array_equal(u.latitude.values, other.latitude.values) or not np.array_equal(u.longitude.values, other.longitude.values):
            raise ValueError("GFS GRIB latitude/longitude grids do not match")
    get = lambda da, lev: da.sel(isobaricInhPa=lev).values
    lon, lat = u.longitude.values, u.latitude.values
    diag = calculate_diagnostics(
        lon, lat, get(u, 250), get(v, 250), get(u, 500), get(v, 500),
        get(w, 400), get(w, 500), get(w, 600)
    )
    return lon, lat, diag


def render(lon, lat, diagnostic, cycle, hour, output=PLOT_PATH):
    """Three independent fields, with coordinate axes and Kuwait locator."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    descriptions = (
        ("div250_1e5_s", "250 hPa wind divergence (10⁻⁵ s⁻¹)",
         "RdBu_r", -10, 10),
        ("upward_omega_pa_s", "600–400 hPa mean upward omega (Pa/s; + = ascent)",
         "RdBu", -.8, .8),
        ("abs_vort500_1e5_s", "500 hPa absolute vorticity (10⁻⁵ s⁻¹)",
         "RdBu_r", -15, 15),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    valid = cycle + timedelta(hours=hour)
    for ax, (key, label, cmap, vmin, vmax) in zip(axes, descriptions):
        values = diagnostic[key]
        mesh = ax.pcolormesh(lon, lat, values, shading="auto", cmap=cmap,
                             vmin=vmin, vmax=vmax)
        ax.plot(47.98, 29.38, "k*", ms=10)
        ax.annotate("Kuwait", (47.98, 29.38), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, weight="bold")
        ax.set_xlim(PLOT_BOUNDS[:2])
        ax.set_ylim(PLOT_BOUNDS[2:])
        ax.grid(alpha=.25)
        ax.set_xlabel("Longitude °E")
        ax.set_ylabel("Latitude °N")
        ax.set_title(label, fontsize=10)
        fig.colorbar(mesh, ax=ax, shrink=.72, pad=.02)
    fig.suptitle(
        f"EXPERIMENTAL GFS Upper-Air Dynamics | Run {cycle:%Y-%m-%d %HZ} "
        f"| +{hour}h | Valid {valid:%Y-%m-%d %HZ}\n"
        "Model diagnostics only — not PV, irrotational wind, observed radar or hail probability",
        fontsize=12)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def run(cycle=None, hour=6, source=None, output=PLOT_PATH):
    if source is not None:
        if cycle is None:
            raise ValueError("--cycle required for --source to label valid time")
        grib = Path(source)
        lon, lat, diag = read_diagnostics(grib)
    elif cycle is not None:
        grib = download(cycle, hour)
        lon, lat, diag = read_diagnostics(grib)
    else:
        now = datetime.now(timezone.utc)
        anchor = now.replace(hour=(now.hour // 6) * 6, minute=0,
                             second=0, microsecond=0)
        last_error = None
        for i in range(6):
            candidate = anchor - timedelta(hours=6*i)
            try:
                grib = download(candidate, hour)
                lon, lat, diag = read_diagnostics(grib)
                cycle = candidate
                break
            except (OSError, ValueError, RuntimeError) as exc:
                last_error = exc
                print(f"Skip GFS cycle {candidate:%Y-%m-%d %HZ}: {exc}", flush=True)
        else:
            raise RuntimeError(f"No complete GFS cycle available: {last_error}")
    path = render(lon, lat, diag, cycle, hour, output)
    print(f"Rendered {path} from GFS {cycle:%Y-%m-%d %HZ} +{hour}h", flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", help="GFS model initialization YYYYMMDDHH in UTC")
    parser.add_argument("--hour", type=int, default=6)
    parser.add_argument("--source", help="Use existing NOAA GRIB2 file (offline)")
    parser.add_argument("--output", default=str(PLOT_PATH))
    opts = parser.parse_args()
    c = (datetime.strptime(opts.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
         if opts.cycle else None)
    run(cycle=c, hour=opts.hour, source=opts.source, output=opts.output)
