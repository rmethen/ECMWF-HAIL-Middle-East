"""Meteorologix-style SHIP map rendering, for computed SHIP arrays ONLY.

The plotter does not infer SHIP from lapse/shear proxies. A caller must pass
the independently computed and validated SHIP grid.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

LEVELS = [0, .01, .1, .25, .5, .75, 1, 1.5, 2, 2.5, 3, 4, 10]
COLORS = ["#e5e5e5", "#c4c4c4", "#999999", "#7ba47a",
          "#158a20", "#1397e2", "#8fc7ff", "#ffff31", "#b8ae29",
          "#bc7e1c", "#a80b14", "#f46b97"]


def render_ship(lon, lat, ship, *, cycle_utc, valid_utc,
                output="output/GFS_SHIP_INDEX_PREVIEW.png",
                region=(20, 65, 10, 45)):
    """Render verified SHIP index values, not a hail probability or hail size."""
    ship = np.asarray(ship, dtype=float)
    if ship.shape != np.broadcast_shapes(np.shape(lat), np.shape(lon), ship.shape):
        raise ValueError("Coordinate and SHIP grid dimensions disagree")
    if not np.isfinite(ship).any() or np.nanmin(ship) < 0:
        raise ValueError("SHIP grid must contain nonnegative finite values")
    fig, ax = plt.subplots(figsize=(12, 9), layout="constrained")
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(LEVELS, cmap.N, clip=True)
    im = ax.pcolormesh(lon, lat, ship, shading="auto", cmap=cmap, norm=norm)
    ax.set_xlim(region[0], region[1])
    ax.set_ylim(region[2], region[3])
    ax.set_xlabel("Longitude E")
    ax.set_ylabel("Latitude N")
    ax.grid(alpha=.2)
    ax.set_title("GFS SHIP — Significant Hail Parameter (index)\n"
                 f"Run: {cycle_utc} UTC   Valid: {valid_utc} UTC", weight="bold")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=.07,
                      ticks=LEVELS[1:-1], extend="max")
    cb.set_label("SHIP index (NOT probability, NOT guaranteed hail size)")
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    plt.close(fig)
    return dest
