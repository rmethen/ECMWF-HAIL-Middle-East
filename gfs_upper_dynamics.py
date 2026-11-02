"""Experimental GFS upper-air dynamics for Kuwait and the Gulf.

Independent of operational ECMWF and GFS hail indices.
Computes 250-hPa horizontal divergence, 600–400-hPa mean upward motion,
and 500-hPa absolute vorticity from *real* NOAA GFS pressure-level fields.

This is NOT the Bentley irrotational-wind or Ertel-PV product, and NOT radar.
"""
from __future__ import annotations

import argparse
import json
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
        "rel_vort500_1e5_s": rel_vort500 * 1e5,
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


ASCENT_CORE_THRESHOLD_PA_S = 0.50
ASCENT_CORE_NEIGHBORHOOD = 9


def find_ascent_cores(upward_omega, threshold=ASCENT_CORE_THRESHOLD_PA_S,
                      neighborhood=ASCENT_CORE_NEIGHBORHOOD):
    """Unique local maxima of *raw* upward omega, displayed as ascent dots.

    Positive values represent ascent because GFS pressure velocity is negated.
    A connected flat maximum (plateau) gives exactly one representative dot.
    These are *model-derived ascent peaks*, not observed precipitation or hail.
    """
    from scipy.ndimage import label, maximum_filter

    field = np.asarray(upward_omega, dtype=float)
    if field.ndim != 2:
        raise ValueError("upward_omega must be 2-D")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite (Pa/s)")
    if not isinstance(neighborhood, int) or neighborhood < 3 or neighborhood % 2 == 0:
        raise ValueError("neighborhood must be odd and >= 3")
    finite = np.isfinite(field)
    work = np.where(finite, field, -np.inf)
    local_max = maximum_filter(work, size=neighborhood, mode="nearest")
    candidates = finite & (work >= threshold) & (work == local_max)
    components, count = label(candidates, structure=np.ones((3, 3), dtype=int))
    cores = np.zeros(field.shape, dtype=bool)
    for region in range(1, count + 1):
        points = np.argwhere(components == region)
        centroid = points.mean(axis=0)
        i = np.argmin(((points - centroid) ** 2).sum(axis=1))
        row, col = points[i]
        cores[row, col] = True
    return cores


def render(lon, lat, diagnostic, cycle, hour, output=PLOT_PATH):
    """Geographically grounded three-panel synoptic chart.

    Display-only Gaussian filter (~0.75 degree sigma on 0.25-degree grid)
    reduces single-cell derivative noise; the scientific outputs remain raw.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    fields = (
        ("div250_1e5_s", "250 hPa full-wind divergence (10⁻⁵ s⁻¹)",
         "RdBu_r", -10, 10),
        ("upward_omega_pa_s", "600–400 hPa mean upward omega (Pa/s; + = ascent)",
         "RdBu", -.8, .8),
        ("rel_vort500_1e5_s", "500 hPa relative vorticity (10⁻⁵ s⁻¹)",
         "RdBu_r", -10, 10),
    )
    crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, 3, figsize=(18.5, 7),
        subplot_kw={"projection": crs}, constrained_layout=True)
    valid = cycle + timedelta(hours=hour)
    places = (
        ("Kuwait", 47.98, 29.38),
        ("Basra", 47.78, 30.51),
        ("Riyadh", 46.71, 24.71),
        ("Doha", 51.53, 25.29),
        ("Dubai", 55.27, 25.20),
        ("Muscat", 58.59, 23.61),
    )
    # 0.25 degree regular GFS grid; only *plotted* values are smoothed.
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    step_lat = float(np.median(np.abs(np.diff(lat))))
    step_lon = float(np.median(np.abs(np.diff(lon))))
    sigma = (0.75 / step_lat, 0.75 / step_lon)
    for ax, (key, label, cmap, vmin, vmax) in zip(axes, fields):
        values = np.asarray(diagnostic[key], dtype=float)
        if values.shape != (len(lat), len(lon)):
            raise ValueError(f"Invalid diagnostic shape for {key}")
        if not np.isfinite(values).all():
            # Normalized Gaussian convolution prevents spreading missing points.
            weights = gaussian_filter(np.isfinite(values).astype(float), sigma=sigma)
            smoothed = gaussian_filter(np.nan_to_num(values, nan=0., posinf=0., neginf=0.),
                                       sigma=sigma) / np.maximum(weights, 1e-10)
            smoothed[weights < 0.05] = np.nan
        else:
            smoothed = gaussian_filter(values, sigma=sigma)
        mesh = ax.pcolormesh(
            lon, lat, smoothed, shading="auto", cmap=cmap,
            vmin=vmin, vmax=vmax, transform=crs, rasterized=True)
        # Dots use unsmoothed source-grid peaks, so localized forcing is not
        # lost to the 0.75-degree smoothing applied to background shading.
        if key == "upward_omega_pa_s":
            core_mask = find_ascent_cores(values)
            lon2d, lat2d = np.meshgrid(lon, lat)
            west, east, south, north = PLOT_BOUNDS
            visible = (core_mask & (lon2d >= west) & (lon2d <= east) &
                       (lat2d >= south) & (lat2d <= north))
            if np.any(visible):
                ax.scatter(
                    lon2d[visible], lat2d[visible],
                    s=22, c="#df0027", marker="o",
                    edgecolors="white", linewidths=0.55,
                    transform=crs, zorder=4,
                )
            ax.text(
                0.02, 0.025,
                f"Red dots: raw ascent cores ≥ {ASCENT_CORE_THRESHOLD_PA_S:.2f} Pa/s "
                f"({np.count_nonzero(visible)})",
                transform=ax.transAxes, fontsize=7, zorder=7,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "alpha": 0.86, "pad": 2.5},
            )
        ax.set_extent(PLOT_BOUNDS, crs=crs)
        # Public-domain Natural Earth outlines; no arbitrary drawn coastlines.
        ax.coastlines(resolution="110m", color="#263647", linewidth=0.75, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"),
                       edgecolor="#3d4751", linewidth=0.5, zorder=3)
        ax.set_xticks(np.arange(35, 65, 5), crs=crs)
        ax.set_yticks(np.arange(16, 39, 4), crs=crs)
        ax.tick_params(labelsize=8)
        ax.grid(color="#7b8490", alpha=.25, linewidth=.5)
        ax.set_xlabel("Longitude °E")
        ax.set_ylabel("Latitude °N")
        for name, x, y in places:
            if name == "Kuwait":
                ax.plot(x, y, marker="*", color="black", markersize=8,
                        transform=crs, zorder=5)
            else:
                ax.plot(x, y, marker="o", color="black", markersize=2.2,
                        transform=crs, zorder=5)
            offset = ((-38, -13) if name == "Kuwait" else
                      (4, 9) if name == "Basra" else (4, 4))
            ax.annotate(name, xy=(x, y), xytext=offset,
                        textcoords="offset points", fontsize=7, weight="bold",
                        color="black",
                        bbox={"facecolor": "white", "edgecolor": "none",
                              "alpha": .65, "pad": .6}, zorder=6)
        ax.set_title(label, fontsize=10)
        fig.colorbar(mesh, ax=ax, shrink=.77, pad=.025, extend="both")
    fig.suptitle(
        f"EXPERIMENTAL GFS Upper-Air Dynamics | Run {cycle:%Y-%m-%d %HZ} "
        f"| +{hour}h | Valid {valid:%Y-%m-%d %HZ}\n"
        "Display smoothed ~0.75° sigma | Independent diagnostics, not PV, irrotational wind or hail probability",
        fontsize=12)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
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


def render_ascent_focus(lon, lat, diagnostic, cycle, hour, output):
    """Large single-panel Gulf ascent image intended to be readable on a phone.

    Original three-panel diagnostics remain available. All annotations refer
    to *forecast* pressure-vertical motion, not actual rainfall observations.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from scipy.ndimage import gaussian_filter

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    omega = np.asarray(diagnostic["upward_omega_pa_s"], dtype=float)
    if omega.shape != (len(lat), len(lon)):
        raise ValueError("Omega must be on the GFS lat/lon grid")

    # Reduce display noise modestly; dots and location labels use RAW field.
    sigmas = (0.3 / np.median(np.abs(np.diff(lat))),
              0.3 / np.median(np.abs(np.diff(lon))))
    weights = gaussian_filter(np.isfinite(omega).astype(float), sigmas)
    shaded = gaussian_filter(np.where(np.isfinite(omega), omega, 0.0), sigmas) \
             / np.maximum(weights, 1e-10)
    shaded[weights < 0.2] = np.nan

    crs = ccrs.PlateCarree()
    # Aspect ratio appropriate to Kuwait/Gulf at page width on mobile.
    fig = plt.figure(figsize=(9, 10.5), facecolor="white")
    ax = fig.add_axes([0.07, 0.20, 0.81, 0.63], projection=crs)
    extent = (42, 59, 22, 36)
    ax.set_extent(extent, crs=crs)
    cmap = plt.get_cmap("RdBu_r")  # Red = rising (+); blue = sinking (-).
    image = ax.pcolormesh(
        lon, lat, shaded, cmap=cmap,
        norm=TwoSlopeNorm(vmin=-0.65, vcenter=0, vmax=0.65),
        transform=crs, shading="auto", rasterized=True)
    # Strong-ascent extent, not a precipitation amount or probability.
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)
    if np.nanmax(shaded) >= 0.25:
        ax.contour(lon, lat, shaded, levels=[0.25],
                   colors=["#a71930"], linewidths=1.3,
                   linestyles="dashed", transform=crs)
    if np.nanmax(shaded) >= 0.50:
        ax.contour(lon, lat, shaded, levels=[0.50],
                   colors=["#900016"], linewidths=2.2, transform=crs)

    core_mask = find_ascent_cores(omega)
    west, east, south, north = extent
    shown = core_mask & (mesh_lon >= west) & (mesh_lon <= east) & \
            (mesh_lat >= south) & (mesh_lat <= north)
    coords = np.argwhere(shown)
    if len(coords):
        ax.scatter(
            mesh_lon[shown], mesh_lat[shown], marker="o",
            s=88, c="#d90c23", edgecolors="#ffe144", linewidths=1.9,
            transform=crs, zorder=8)
        # Label only strongest three values to avoid unreadable clusters.
        sorted_coords = sorted(coords, key=lambda ij: omega[tuple(ij)], reverse=True)
        for i, j in sorted_coords[:3]:
            ax.annotate(f"{omega[i, j]:.2f}",
                        (lon[j], lat[i]), xytext=(7, 8),
                        textcoords="offset points", size=10, weight="bold",
                        color="#8e0018",
                        bbox={"facecolor": "white", "edgecolor": "none",
                              "alpha": 0.85, "pad": 1.5}, zorder=9)

    ax.coastlines(resolution="110m", linewidth=1.1, color="#263344", zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"),
                   linewidth=0.75, edgecolor="#404850", zorder=4)
    for name, x, y in (
        ("Basra", 47.78, 30.51), ("KUWAIT", 47.98, 29.38),
        ("Doha", 51.53, 25.29), ("Dubai", 55.27, 25.20),
        ("Riyadh", 46.71, 24.71), ("Bushehr", 50.84, 28.92),
    ):
        ax.plot(x, y, "k*" if name == "KUWAIT" else "ko",
                markersize=10 if name == "KUWAIT" else 3.5,
                transform=crs, zorder=7)
        dx, dy = ((-52, -18) if name == "KUWAIT" else
                  (-40, 11) if name == "Basra" else (5, 6))
        ax.annotate(name, (x, y), xytext=(dx, dy),
                    textcoords="offset points", size=10,
                    weight="bold" if name == "KUWAIT" else "normal",
                    color="#1b2631",
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "alpha": 0.78, "pad": 0.8}, zorder=9)
    ax.set_xticks(np.arange(42, 60, 3), crs=crs)
    ax.set_yticks(np.arange(22, 37, 2), crs=crs)
    ax.tick_params(labelsize=10)
    ax.grid(color="#607283", alpha=.22)

    valid = cycle + timedelta(hours=hour)
    valid_kwt = valid + timedelta(hours=3)
    fig.text(.07, .962, "GFS | KUWAIT & NORTH GULF", fontsize=21,
             weight="bold", color="#172b40")
    fig.text(.07, .925, "600–400 hPa vertical motion | RED = RISING AIR",
             fontsize=13, weight="bold", color="#a40c25")
    fig.text(.07, .895,
             f"Cycle {cycle:%Y-%m-%d %HZ}    Forecast +{hour}h"
             f"    Valid Kuwait {valid_kwt:%d %b %H:%M}",
             fontsize=11, color="#374656")

    cb = fig.colorbar(image, ax=ax, fraction=.052, pad=.022, shrink=.81)
    cb.ax.tick_params(labelsize=10)
    cb.set_label("Upward omega (Pa/s)", fontsize=11)
    fig.text(.075, .155,
             f"RED/YELLOW dots: {len(coords)} local maxima of ascent >= 0.50 Pa/s.",
             fontsize=12, weight="bold", color="#a40c25")
    fig.text(.075, .123,
             "Red outlines: 0.25 (dashed) / 0.50 (solid) Pa/s ascent.",
             fontsize=11, color="#3c4b5a")
    fig.text(.075, .087,
             "FORECAST DYNAMICS ONLY. Not observed storms, rain or hail.",
             fontsize=11, weight="bold", color="#36475a")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return output


def run_hours(cycle=None, hours=(3, 6, 12), output=PLOT_PATH, status_output=None, ascent_focus=False):
    """Render multiple leads from ONE complete GFS initialization.

    Cycle rollback is all-or-none so the evolution has consistent run time.
    """
    hours = tuple(int(h) for h in hours)
    if not hours or len(hours) > 5 or len(set(hours)) != len(hours):
        raise ValueError("Need 1–5 distinct GFS forecast hours")
    if any(h < 0 or h > 120 for h in hours):
        raise ValueError("All forecast hours must be 0–120")
    if cycle is None:
        now = datetime.now(timezone.utc)
        anchor = now.replace(hour=now.hour // 6 * 6, minute=0, second=0,
                             microsecond=0)
        candidates = [anchor - timedelta(hours=6 * i) for i in range(6)]
    else:
        candidates = [cycle]

    last_error = None
    for candidate in candidates:
        sequence = []
        try:
            for hour in hours:
                path = Path(f"data/gfs_upper_dynamics_f{hour:03d}.grib2")
                grib = download(candidate, hour, path=path)
                lon, lat, diag = read_diagnostics(grib)
                sequence.append((hour, lon, lat, diag))
            break
        except (OSError, ValueError, RuntimeError) as exc:
            last_error = exc
            print(f"Reject incomplete GFS cycle {candidate:%Y-%m-%d %HZ}: "
                  f"{exc}", flush=True)
    else:
        raise RuntimeError(f"No complete GFS cycle for {hours}: {last_error}")

    out = Path(output)
    paths = []
    for hour, lon, lat, diag in sequence:
        destination = out.with_name(f"{out.stem}_F{hour:03d}{out.suffix}")
        result = render(lon, lat, diag, candidate, hour, destination)
        paths.append(result)
        if ascent_focus:
            focus_path = out.with_name(f"GFS_ASCENT_CORES_GULF_F{hour:03d}.png")
            render_ascent_focus(lon, lat, diag, candidate, hour, focus_path)
            print(f"Rendered mobile ascent focus: {focus_path}", flush=True)
        print(f"Rendered real GFS cycle {candidate:%Y-%m-%d %HZ} +{hour}h: "
              f"{result}", flush=True)
    # Write status only after all three real-data images exist; a failed run
    # must leave the previously published cycle and image timestamps intact.
    if status_output is not None:
        status_path = Path(status_output)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        # Never publish a fallback run older than the currently visible cycle.
        if status_path.exists():
            try:
                previous = json.loads(status_path.read_text(encoding="utf-8"))
                last_cycle = datetime.fromisoformat(
                    previous["cycle_utc"].replace("Z", "+00:00"))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid existing GFS publication status: {exc}") from exc
            if candidate < last_cycle:
                raise ValueError(
                    f"Refusing to replace newer GFS {last_cycle.isoformat()} "
                    f"with older fallback {candidate.isoformat()}")
        status = {
            "model": "NOAA GFS 0.25 degree",
            "cycle_utc": candidate.isoformat().replace("+00:00", "Z"),
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "forecast_hours": list(hours),
            "valid_utc": {str(h): (candidate + timedelta(hours=h)).isoformat().replace("+00:00", "Z")
                          for h in hours},
            "ascent_core_threshold_pa_s": ASCENT_CORE_THRESHOLD_PA_S,
            "ascent_local_max_window_grid_points": ASCENT_CORE_NEIGHBORHOOD,
            "product": "Experimental upper-air ascent cores (not observed rain)",
        }
        temporary = status_path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
            temporary.replace(status_path)
        finally:
            temporary.unlink(missing_ok=True)
    return paths



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", help="GFS model initialization YYYYMMDDHH in UTC")
    parser.add_argument("--hour", type=int, default=6)
    parser.add_argument("--hours", help="Comma-separated lead hours, e.g. 3,6,12, one GFS cycle")
    parser.add_argument("--source", help="Use existing NOAA GRIB2 file (offline)")
    parser.add_argument("--output", default=str(PLOT_PATH))
    parser.add_argument("--status-output", help="Write JSON cycle metadata after all lead-time images succeed")
    parser.add_argument("--ascent-focus", action="store_true", help="Also render a large single-panel ascent map for phones")
    opts = parser.parse_args()
    c = (datetime.strptime(opts.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
         if opts.cycle else None)
    if opts.hours:
        if opts.source:
            parser.error("--source is only supported with --hour (single lead)")
        run_hours(cycle=c, hours=tuple(int(h.strip()) for h in opts.hours.split(",")),
                  output=opts.output, status_output=opts.status_output,
                  ascent_focus=opts.ascent_focus)
    else:
        run(cycle=c, hour=opts.hour, source=opts.source, output=opts.output)
