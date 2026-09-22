"""Operational GFS VP200 forecast at Kelvin-wave spatial scales."""

from datetime import timedelta
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

import vp200_mjo_scale as base


DATA_DIR = Path("data/vp200_kelvin_scale")
OUT_FILE = Path("output/VP200_KELVIN_SCALE_FORECAST_LATEST.png")


def kelvin_spatial_filter(field):
    """Keep the equatorially symmetric, zonal-wave 1-14 component."""
    symmetric = 0.5 * (field + np.flip(field, axis=0))
    spectrum = np.fft.rfft(symmetric, axis=1)
    filtered = np.zeros_like(spectrum)
    filtered[:, 1:15] = spectrum[:, 1:15]
    return np.fft.irfft(filtered, n=field.shape[1], axis=1)


def plot(fields, lat, lon, cycle):
    panels = [
        (fields[0], "Today", cycle),
        (np.mean([fields[h] for h in range(24, 97, 24)], axis=0),
         "Days 1-4 mean", cycle + timedelta(days=2)),
        (np.mean([fields[h] for h in range(120, 193, 24)], axis=0),
         "Days 5-8 mean", cycle + timedelta(days=6)),
    ]
    levels = np.arange(-6, 7, 1)
    colors = ["#a500db", "#4800bd", "#0035ba", "#1971d4", "#79abe6", "#d7e9f6",
              "#ffffff", "#fff1ac", "#ffc955", "#f68b27", "#e64517", "#bd0909"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, cmap.N)
    projection = ccrs.PlateCarree(central_longitude=180)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11.5), subplot_kw={"projection": projection})
    mesh = None
    for ax, (field, label, valid) in zip(axes, panels):
        mesh = ax.contourf(lon, lat, field / 1e6, levels=levels, cmap=cmap, norm=norm,
                           extend="both", transform=ccrs.PlateCarree())
        ax.contour(lon, lat, field / 1e6, levels=levels, colors="black", linewidths=0.25,
                   alpha=0.4, transform=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)
        ax.set_extent([0, 360, -30, 30], crs=ccrs.PlateCarree())
        ax.set_title(f"{label}  |  centred {valid:%d %b %Y}", loc="left", fontsize=12)
        grid = ax.gridlines(draw_labels=True, linewidth=0.2, color="0.45", alpha=0.3)
        grid.top_labels = grid.right_labels = False
    fig.suptitle("Kelvin-scale VP200 Forecast (Symmetric, Zonal Waves 1-14)", fontsize=18, y=0.99)
    cbar = fig.colorbar(mesh, ax=axes, orientation="horizontal", pad=0.04,
                        fraction=0.04, ticks=np.arange(-6, 7, 2))
    cbar.set_label("VP200 anomaly [10⁶ m² s⁻¹]   Blue/purple: upper divergence | Red: convergence")
    fig.text(0.5, 0.012,
             "GFS | 1991-2020 NCEP/NCAR climatology | spatial Kelvin-scale guide; not a full time-space filter",
             ha="center", fontsize=9)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    base.DATA_DIR = DATA_DIR
    base.OUT_FILE = OUT_FILE
    base.planetary_filter = kelvin_spatial_filter
    base.plot = plot
    base.main()


if __name__ == "__main__":
    main()
