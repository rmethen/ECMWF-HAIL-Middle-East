"""Generate the ECMWF Thunderstorm & Lightning Potential map."""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from thunderstorm_index import thunderstorm_potential

DATA_FILE = Path("data/hail_diagnostics.npz")
OUTPUT_FILE = Path("output/ECMWF_THUNDERSTORM_LIGHTNING_POTENTIAL_LATEST.png")


def smooth_field(field):
    """Apply two light passes to reduce pixel noise without erasing signals."""
    result = np.asarray(field, dtype=float)
    for _ in range(2):
        padded = np.pad(result, 1, mode="edge")
        result = sum(
            padded[i:i + result.shape[0], j:j + result.shape[1]]
            for i in range(3) for j in range(3)
        ) / 9.0
    return result


def white_low_cmap():
    colors = plt.get_cmap("turbo", 10)(np.arange(10))
    colors[0] = (1.0, 1.0, 1.0, 1.0)
    return ListedColormap(colors)


def format_time(value):
    return np.datetime_as_string(np.datetime64(value, "m"), unit="m").replace("T", " ") + " UTC"


def main():
    data = np.load(DATA_FILE)
    index = thunderstorm_potential(
        data["li850"], data["omega_700"], data["omega_500"], data["moisture_850"],
        data["lapse_700_500"], data["shear_850_300"], data["t500_c"],
    )
    field = np.nanmax(index, axis=0) if index.ndim == 3 else index
    field = smooth_field(field)
    step_scores = np.nanmax(index.reshape(index.shape[0], -1), axis=1)
    peak_idx = int(np.nanargmax(step_scores))
    step = data["steps"][peak_idx]
    peak_hour = (
        int(step / np.timedelta64(1, "h"))
        if np.issubdtype(step.dtype, np.timedelta64)
        else int(step)
    )
    init_time = data["init_time"]
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([20, 65, 10, 45], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    ax.add_feature(cfeature.LAND, facecolor="white")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    display_field = np.ma.masked_less(field, 15.0)
    plot = ax.contourf(
        data["longitude"], data["latitude"], display_field,
        levels=np.arange(0, 101, 10), cmap=white_low_cmap(), extend="max",
        transform=ccrs.PlateCarree(),
    )
    ax.plot(47.98, 29.38, marker="*", color="black", markersize=10,
            transform=ccrs.PlateCarree())
    ax.text(48.3, 29.5, "Kuwait", fontsize=9, transform=ccrs.PlateCarree())
    z500_peak = data["z500_m"][peak_idx]
    zmin = np.floor(np.nanmin(z500_peak) / 60.0) * 60.0
    zmax = np.ceil(np.nanmax(z500_peak) / 60.0) * 60.0
    if zmax > zmin:
        heights = ax.contour(
            data["longitude"], data["latitude"], z500_peak,
            levels=np.arange(zmin, zmax + 1.0, 60.0), colors="#666666",
            linewidths=0.45, alpha=0.55, transform=ccrs.PlateCarree(),
        )
        ax.clabel(heights, inline=True, fontsize=6, fmt="%d")
    grid = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5)
    grid.top_labels = False
    grid.right_labels = False
    cbar = plt.colorbar(plot, ax=ax, pad=0.025, shrink=0.82)
    cbar.set_label("Thunderstorm & Lightning Potential (0–100)")
    plt.title(
        "ECMWF Experimental Thunderstorm & Lightning Potential – Middle East\n"
        f"Init: {format_time(init_time)} | Peak: +{peak_hour} h | "
        f"Valid: {format_time(init_time + np.timedelta64(peak_hour, 'h'))}",
        fontsize=14, weight="bold",
    )
    plt.figtext(
        0.5, 0.02,
        "Storm support: LI • Omega 700/500 • Moisture • Lapse • Shear | Gray: 500-hPa height (m)",
        ha="center", fontsize=9,
    )
    plt.figtext(
        0.94, 0.02, f"Max: {np.nanmax(field):.1f}",
        ha="right", fontsize=10, weight="bold",
    )
    plt.savefig(OUTPUT_FILE, dpi=160, bbox_inches="tight")
    plt.close()
    print("Map saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
