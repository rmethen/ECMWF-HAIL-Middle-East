"""Regional Helmholtz potential-wind approximation on a limited GFS lat/lon grid.

This is a *bounded-domain* Hodge/Helmholtz gradient component with zero
velocity-potential boundary values, not a global spherical-harmonic Helmholtz
decomposition such as the Albany diagnostic. Result depends on domain and
boundary assumptions. Use an enlarged NOAA domain and plot its interior only.
"""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def regional_irrotational_wind(lon, lat, u, v):
    """Approximate 250-hPa divergent-component wind from the local wind field.

    Invert the spherical scalar Poisson equation

        ∇²_sphere(chi) = div_sphere(horizontal_wind)

    with chi=0 on the outer rectangular lat/lon boundary. Returns the
    irrotational potential-gradient component (u_chi, v_chi), restored to the
    original latitude and longitude order. Units are m/s. Strictly REGIONAL,
    not equivalent to global irrotational wind without global boundary data.

    The numerical solver uses a 5-point latitude-weighted spherical Laplacian
    and scipy's sparse linear solver; no raw full-wind arrows are substituted.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if lon.ndim != 1 or lat.ndim != 1 or len(lon) < 7 or len(lat) < 7:
        raise ValueError("Need at least 7 coordinates on each regular grid axis")
    if u.shape != (len(lat), len(lon)) or v.shape != u.shape:
        raise ValueError("Input u/v must match latitude x longitude dimensions")
    if not all(np.isfinite(x).all() for x in (lon, lat, u, v)):
        raise ValueError("Irrotational wind requires finite coordinates and winds")
    if np.max(np.abs(lat)) >= 80:
        raise ValueError("Bounded solver is not intended for polar grids")

    lon_diff, lat_diff = np.diff(lon), np.diff(lat)
    if not ((lon_diff > 0).all() or (lon_diff < 0).all()):
        raise ValueError("longitude must be monotonic")
    if not ((lat_diff > 0).all() or (lat_diff < 0).all()):
        raise ValueError("latitude must be monotonic")
    flip_lon, flip_lat = lon_diff[0] < 0, lat_diff[0] < 0
    if flip_lon:
        lon, u, v = lon[::-1], u[:, ::-1], v[:, ::-1]
    if flip_lat:
        lat, u, v = lat[::-1], u[::-1, :], v[::-1, :]

    lons, lats = np.deg2rad(lon), np.deg2rad(lat)
    dl = float(np.median(np.diff(lons)))
    dp = float(np.median(np.diff(lats)))
    if not np.allclose(np.diff(lons), dl, rtol=1e-4) or not np.allclose(
        np.diff(lats), dp, rtol=1e-4
    ):
        raise ValueError("Helmholtz solver requires a regular lat/lon grid")

    cos = np.cos(lats)
    # Physical spherical divergence of the full wind (s^-1).
    div = (np.gradient(u, lons, axis=1, edge_order=2) +
           np.gradient(v * cos[:, None], lats, axis=0, edge_order=2)
           ) / (EARTH_RADIUS_M * cos[:, None])

    # Set chi = EARTH_RADIUS_M**2 * q, yielding:
    # (1/cos(phi)) [d/dphi(cos(phi) dq/dphi) +
    #                 (1/cos(phi)) d²q/dlambda²] = divergence.
    # A positive definite discretization of -L is used for sparse solve.
    ny, nx = len(lat), len(lon)
    ni, nj = ny - 2, nx - 2
    count = ni * nj
    cp = cos[1:-1]
    east = 1.0 / ((cp * dl) ** 2)
    north = np.cos(lats[1:-1] + dp / 2) / (cp * dp ** 2)
    south = np.cos(lats[1:-1] - dp / 2) / (cp * dp ** 2)
    ew = np.broadcast_to(east[:, None], (ni, nj)).ravel().copy()
    nn = np.broadcast_to(north[:, None], (ni, nj)).ravel().copy()
    ss = np.broadcast_to(south[:, None], (ni, nj)).ravel().copy()
    diag = 2 * ew + nn + ss

    # Stop ±1 diagonals from coupling the end of one latitude row to
    # the beginning of another.
    eoff = ew[:-1].copy()
    eoff[(np.arange(count - 1) + 1) % nj == 0] = 0
    woff = ew[1:].copy()
    woff[np.arange(1, count) % nj == 0] = 0
    mat = diags([diag, -eoff, -woff, -nn[:-nj], -ss[nj:]],
                offsets=[0, 1, -1, nj, -nj], format="csr")
    q = np.zeros((ny, nx), dtype=float)
    q[1:-1, 1:-1] = spsolve(mat, -div[1:-1, 1:-1].ravel()).reshape((ni, nj))
    if not np.isfinite(q).all():
        raise ValueError("Regional Helmholtz Poisson solver did not converge")

    # For a potential chi, uchi = dchi/dx and vchi = dchi/dy.
    uchi = EARTH_RADIUS_M * np.gradient(q, lons, axis=1,
                                       edge_order=2) / cos[:, None]
    vchi = EARTH_RADIUS_M * np.gradient(q, lats, axis=0,
                                       edge_order=2)
    if flip_lat:
        uchi, vchi = uchi[::-1, :], vchi[::-1, :]
    if flip_lon:
        uchi, vchi = uchi[:, ::-1], vchi[:, ::-1]
    return uchi, vchi
