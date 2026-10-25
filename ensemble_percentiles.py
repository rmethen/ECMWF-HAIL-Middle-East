"""Pointwise precipitation percentiles shared by the ensemble map products."""

from pathlib import Path

import numpy as np


PRECIP_LEVELS = [1, 5, 10, 25, 50, 75, 100, 150, 200]
PERCENTILES = (80, 90)


def precipitation_percentiles(rain, percentiles=PERCENTILES):
    """Return linear quantiles along the member axis, in mm at each grid point.

    Each member must represent the same accumulation period and grid. Reject
    missing values rather than silently using fewer members at some locations.
    """
    rain = np.asarray(rain)
    if rain.ndim != 3 or rain.shape[0] < 2 or 0 in rain.shape:
        raise ValueError("Expected precipitation with member, latitude, longitude axes")
    if not np.isfinite(rain).all() or (rain < 0).any():
        raise ValueError("Ensemble precipitation must be finite and nonnegative")
    return np.percentile(rain, percentiles, axis=0, method="linear")


def plot_percentiles(rain, lats, lons, init, model, output_prefix, base_ax):
    """Draw P80/P90 using the wettest-member map's grid, period and colour scale."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    fields = precipitation_percentiles(rain)
    paths = []
    for percentile, field in zip(PERCENTILES, fields):
        target = Path(f"output/{output_prefix}_P{percentile}_0_72H_LATEST.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = base_ax(
            f"{model} P{percentile}: 0–72 h Accumulated Precipitation\n"
            f"Init: {init} UTC | {rain.shape[0]} members"
        )
        plot = ax.contourf(
            lons, lats, np.ma.masked_less(field, 1.0),
            levels=PRECIP_LEVELS, cmap="turbo", extend="max",
            transform=ccrs.PlateCarree(),
        )
        cbar = plt.colorbar(plot, ax=ax, pad=0.025, shrink=0.82)
        cbar.set_label("Accumulated precipitation (mm)")
        plt.figtext(
            0.5, 0.02,
            f"Pointwise P{percentile}: about {percentile}% of members at or below; "
            f"{100 - percentile}% above\n"
            "Ensemble rainfall quantile, not rain probability or a climate-extreme index",
            ha="center", fontsize=10,
        )
        fig.savefig(target, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(target)
    return paths
